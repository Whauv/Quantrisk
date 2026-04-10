"""Pipeline orchestration utilities for Quantrisk."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

RESULT_ARTIFACTS: tuple[str, ...] = (
    "feature_matrix",
    "regime_labels",
    "regime_probabilities",
    "risk_metrics",
    "scenario_results",
)


def validate_date_window(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a requested pipeline date window."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("`start_date` must be earlier than `end_date`.")
    return start, end


def normalize_portfolio_weights(portfolio_weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize portfolio weights so they sum to one."""
    if not portfolio_weights:
        raise ValueError("`portfolio_weights` cannot be empty.")

    weights = pd.Series(portfolio_weights, dtype=float)
    if not weights.map(pd.notna).all():
        raise ValueError("Portfolio weights must be finite numeric values.")

    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("Portfolio weights must sum to a non-zero value.")

    if abs(total - 1.0) > 1e-9:
        LOGGER.info("Normalizing portfolio weights because they do not sum to 1.0.")
        weights = weights / total

    return weights.to_dict()


def get_data_dir() -> Path:
    """Return the local data directory used for pipeline artifacts."""
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_artifact_paths(data_dir: Path | None = None) -> dict[str, Path]:
    """Return canonical artifact paths for persisted pipeline outputs."""
    resolved_dir = data_dir if data_dir is not None else get_data_dir()
    return {
        "main_data": resolved_dir / "main_with_regimes.parquet",
        "returns": resolved_dir / "returns.parquet",
        "risk_metrics": resolved_dir / "risk_metrics.parquet",
        "scenario_results": resolved_dir / "scenario_results.parquet",
        "feature_matrix": resolved_dir / "feature_matrix.parquet",
        "regime_labels": resolved_dir / "regime_labels.parquet",
        "regime_probabilities": resolved_dir / "regime_probabilities.parquet",
        "regime_output": resolved_dir / "regime_output.parquet",
        "ingested_data": resolved_dir / "ingested_data.parquet",
    }


def load_cached_results(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load previously saved result artifacts from local Parquet files."""
    artifact_paths = get_artifact_paths(data_dir)
    missing = [name for name in RESULT_ARTIFACTS if not artifact_paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached artifacts: {missing}")

    return {
        "feature_matrix": pd.read_parquet(artifact_paths["feature_matrix"]),
        "regime_labels": pd.read_parquet(artifact_paths["regime_labels"]),
        "regime_probabilities": pd.read_parquet(artifact_paths["regime_probabilities"]),
        "risk_metrics": pd.read_parquet(artifact_paths["risk_metrics"]),
        "scenario_results": pd.read_parquet(artifact_paths["scenario_results"]),
    }


def run_pipeline(
    start_date: str,
    end_date: str,
    portfolio_weights: dict[str, float],
    n_regimes: int = 4,
) -> dict[str, pd.DataFrame]:
    """Run the end-to-end Quantrisk pipeline and persist intermediate outputs."""
    from quantrisk.features import FeatureEngineer
    from quantrisk.ingestion import DataIngestion
    from quantrisk.regime import RegimeDetector
    from quantrisk.risk import RiskModeler
    from quantrisk.scenario import ScenarioEngine

    validate_date_window(start_date, end_date)
    normalized_weights = normalize_portfolio_weights(portfolio_weights)
    data_dir = get_data_dir()
    artifact_paths = get_artifact_paths(data_dir)

    try:
        ingestion = DataIngestion(
            start_date=start_date,
            end_date=end_date,
            output_path=artifact_paths["ingested_data"],
        )
        main_data = ingestion.save()

        feature_engineer = FeatureEngineer(main_data)
        feature_matrix = feature_engineer.build_feature_matrix()
        feature_matrix.to_parquet(artifact_paths["feature_matrix"])

        price_frame = feature_engineer.get_price_frame()
        returns = price_frame.pct_change().dropna()
        returns.to_parquet(artifact_paths["returns"])

        regime_detector = RegimeDetector(features=feature_matrix, main_data=main_data, n_regimes=n_regimes)
        regime_output = regime_detector.fit()
        regime_output.to_parquet(artifact_paths["regime_output"])

        main_with_regimes = regime_detector.save_to_main_dataframe()
        main_with_regimes.to_parquet(artifact_paths["main_data"])

        risk_modeler = RiskModeler(
            returns=returns,
            regime_data=regime_output,
            portfolio_weights=normalized_weights,
        )
        risk_metrics = risk_modeler.compare_regimes()
        risk_metrics.to_parquet(artifact_paths["risk_metrics"])

        scenario_engine = ScenarioEngine(
            returns=returns,
            regime_data=regime_output,
            portfolio_weights=normalized_weights,
        )
        scenario_results = scenario_engine.run()
        scenario_results.to_parquet(artifact_paths["scenario_results"])

        regime_labels = regime_output[["regime_id", "regime_name"]].copy()
        regime_labels.to_parquet(artifact_paths["regime_labels"])

        regime_probabilities = regime_output.filter(like="regime_prob_").copy()
        regime_probabilities.to_parquet(artifact_paths["regime_probabilities"])

        return {
            "feature_matrix": feature_matrix,
            "regime_labels": regime_labels,
            "regime_probabilities": regime_probabilities,
            "risk_metrics": risk_metrics,
            "scenario_results": scenario_results,
        }
    except Exception as live_error:
        LOGGER.warning("Live pipeline execution failed; attempting to load cached artifacts.", exc_info=live_error)
        try:
            return load_cached_results(data_dir)
        except Exception as cache_error:
            raise RuntimeError(
                "Quantrisk pipeline failed for both live execution and cached artifact fallback."
            ) from cache_error
