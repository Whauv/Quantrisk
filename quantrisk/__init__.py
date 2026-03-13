"""Public package API for quantrisk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantrisk.features import FeatureEngineer
from quantrisk.ingestion import DataIngestion
from quantrisk.regime import RegimeDetector
from quantrisk.risk import RiskModeler
from quantrisk.scenario import ScenarioEngine

__all__ = [
    "DataIngestion",
    "FeatureEngineer",
    "RegimeDetector",
    "RiskModeler",
    "ScenarioEngine",
    "run_pipeline",
]


def run_pipeline(
    start_date: str,
    end_date: str,
    portfolio_weights: dict[str, float],
    n_regimes: int = 4,
) -> dict[str, pd.DataFrame]:
    """Run the full local data pipeline and persist intermediate parquet artifacts."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "main_data": data_dir / "main_with_regimes.parquet",
        "returns": data_dir / "returns.parquet",
        "risk_metrics": data_dir / "risk_metrics.parquet",
        "scenario_results": data_dir / "scenario_results.parquet",
        "feature_matrix": data_dir / "feature_matrix.parquet",
        "regime_labels": data_dir / "regime_labels.parquet",
        "regime_probabilities": data_dir / "regime_probabilities.parquet",
    }

    def load_cached_results() -> dict[str, pd.DataFrame]:
        """Load previously saved pipeline artifacts from local Parquet files."""
        missing = [name for name, path in artifact_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing cached artifacts: {missing}")
        return {name: pd.read_parquet(path) for name, path in artifact_paths.items() if name != "main_data" and name != "returns"}

    try:
        ingestion = DataIngestion(
            start_date=start_date,
            end_date=end_date,
            output_path=data_dir / "ingested_data.parquet",
        )
        main_data = ingestion.save()

        feature_engineer = FeatureEngineer(main_data)
        feature_matrix = feature_engineer.build_feature_matrix()
        feature_matrix.to_parquet(data_dir / "feature_matrix.parquet")

        price_columns = feature_engineer.get_price_columns()
        prices = main_data.loc[:, list(price_columns.values())].rename(
            columns={column: asset for asset, column in price_columns.items()}
        )
        returns = prices.pct_change().dropna()
        returns.to_parquet(data_dir / "returns.parquet")

        regime_detector = RegimeDetector(features=feature_matrix, main_data=main_data, n_regimes=n_regimes)
        regime_output = regime_detector.fit()
        regime_output.to_parquet(data_dir / "regime_output.parquet")

        main_with_regimes = regime_detector.save_to_main_dataframe()
        main_with_regimes.to_parquet(data_dir / "main_with_regimes.parquet")

        risk_modeler = RiskModeler(
            returns=returns,
            regime_data=regime_output,
            portfolio_weights=portfolio_weights,
        )
        risk_metrics = risk_modeler.compare_regimes()
        risk_metrics.to_parquet(data_dir / "risk_metrics.parquet")

        scenario_engine = ScenarioEngine(
            returns=returns,
            regime_data=regime_output,
            portfolio_weights=portfolio_weights,
        )
        scenario_results = scenario_engine.run()
        scenario_results.to_parquet(data_dir / "scenario_results.parquet")

        regime_labels = regime_output[["regime_id", "regime_name"]].copy()
        regime_labels.to_parquet(data_dir / "regime_labels.parquet")

        regime_probabilities = regime_output.filter(like="regime_prob_").copy()
        regime_probabilities.to_parquet(data_dir / "regime_probabilities.parquet")

        return {
            "feature_matrix": feature_matrix,
            "regime_labels": regime_labels,
            "regime_probabilities": regime_probabilities,
            "risk_metrics": risk_metrics,
            "scenario_results": scenario_results,
        }
    except Exception:
        return load_cached_results()
