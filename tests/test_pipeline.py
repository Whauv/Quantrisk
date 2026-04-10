"""Core package tests for Quantrisk."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "yfinance" not in sys.modules:
    yfinance_stub = types.ModuleType("yfinance")
    yfinance_stub.download = lambda *args, **kwargs: pd.DataFrame()
    yfinance_stub.set_tz_cache_location = lambda *args, **kwargs: None
    sys.modules["yfinance"] = yfinance_stub

if "fredapi" not in sys.modules:
    fredapi_stub = types.ModuleType("fredapi")

    class _Fred:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key

        def get_series(self, *args, **kwargs) -> pd.Series:
            return pd.Series(dtype=float)

    fredapi_stub.Fred = _Fred
    sys.modules["fredapi"] = fredapi_stub

from quantrisk import (
    Backtester,
    DataIngestion,
    FeatureEngineer,
    RegimeDetector,
    RiskModeler,
    ScenarioEngine,
    __all__,
    run_pipeline,
)
from quantrisk.pipeline import normalize_portfolio_weights, validate_date_window
from quantrisk.dashboard.charting import apply_zoom_range, build_regime_duration_table, hex_to_rgba


def build_sample_main_data(periods: int = 700) -> pd.DataFrame:
    """Create a deterministic sample market and macro dataset for tests."""
    index = pd.date_range("2018-01-01", periods=periods, freq="D", name="date")
    base_trend = np.linspace(100.0, 180.0, periods)
    wave = np.sin(np.linspace(0.0, 14.0, periods))

    data = pd.DataFrame(
        {
            "sp500_close": base_trend + wave * 2.0,
            "msci_world_close": base_trend * 0.92 + wave * 1.5,
            "vix_close": 20.0 + np.cos(np.linspace(0.0, 20.0, periods)) * 2.5,
            "gold_close": 150.0 + np.sin(np.linspace(0.0, 10.0, periods)) * 3.0,
            "oil_close": 60.0 + np.cos(np.linspace(0.0, 18.0, periods)) * 4.0,
            "eurusd_close": 1.12 + np.sin(np.linspace(0.0, 9.0, periods)) * 0.03,
            "us10y_yield": 2.0 + np.sin(np.linspace(0.0, 6.0, periods)) * 0.25,
            "us2y_yield": 1.6 + np.cos(np.linspace(0.0, 7.0, periods)) * 0.18,
        },
        index=index,
    )
    data["hy_credit_spread"] = 4.0 + np.cos(np.linspace(0.0, 11.0, periods)) * 0.3
    data["ig_credit_spread"] = 1.5 + np.sin(np.linspace(0.0, 8.0, periods)) * 0.15
    return data


class QuantriskTests(unittest.TestCase):
    """End-to-end contract tests for the core Quantrisk modules."""

    def test_public_api_exports_exist(self) -> None:
        """The package should expose the expected public entrypoints."""
        expected = {
            "Backtester",
            "DataIngestion",
            "FeatureEngineer",
            "RegimeDetector",
            "RiskModeler",
            "ScenarioEngine",
            "run_pipeline",
        }
        self.assertEqual(set(__all__), expected)
        self.assertTrue(callable(run_pipeline))
        self.assertEqual(Backtester.__name__, "Backtester")
        self.assertEqual(DataIngestion.__name__, "DataIngestion")
        self.assertEqual(FeatureEngineer.__name__, "FeatureEngineer")
        self.assertEqual(RegimeDetector.__name__, "RegimeDetector")
        self.assertEqual(RiskModeler.__name__, "RiskModeler")
        self.assertEqual(ScenarioEngine.__name__, "ScenarioEngine")

    def test_feature_engineer_builds_clean_matrix(self) -> None:
        """Feature engineering should return a non-empty matrix with no NaNs."""
        feature_engineer = FeatureEngineer(build_sample_main_data())
        feature_matrix = feature_engineer.build_feature_matrix()

        self.assertFalse(feature_matrix.empty)
        self.assertFalse(feature_matrix.isna().any().any())
        self.assertIn("yield_curve_slope", feature_matrix.columns)
        self.assertIn("sp500_vol_20d", feature_matrix.columns)
        self.assertIn("sp500_mom_1m", feature_matrix.columns)

    def test_risk_modeler_compares_regimes(self) -> None:
        """Risk modeling should produce regime-level tail risk metrics."""
        feature_engineer = FeatureEngineer(build_sample_main_data())
        returns = feature_engineer.compute_returns().dropna()

        regime_index = returns.index
        regime_ids = np.where(np.arange(len(regime_index)) % 2 == 0, 0, 1)
        regime_data = pd.DataFrame(
            {
                "regime_id": regime_ids,
                "regime_name": np.where(regime_ids == 0, "Bull", "Bear"),
            },
            index=regime_index,
        )

        model = RiskModeler(
            returns=returns,
            regime_data=regime_data,
            portfolio_weights={
                "sp500": 0.35,
                "msci_world": 0.20,
                "vix": 0.05,
                "gold": 0.15,
                "oil": 0.10,
                "eurusd": 0.15,
            },
        )
        summary = model.compare_regimes()

        self.assertEqual(sorted(summary["regime_id"].tolist()), [0, 1])
        self.assertIn("historical_var_95", summary.columns)
        self.assertIn("parametric_var_99", summary.columns)
        self.assertIn("expected_shortfall_95", summary.columns)
        self.assertIn("average_correlation", summary.columns)

    def test_run_pipeline_uses_cached_results_when_live_execution_fails(self) -> None:
        """The pipeline should fall back to cached artifacts when live execution fails."""
        temp_path = Path("tests") / ".tmp_cache"
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)

        try:
            cached_payload = {
                "feature_matrix": pd.DataFrame({"feature": [1.0, 2.0]}),
                "regime_labels": pd.DataFrame({"regime_id": [0], "regime_name": ["Bull"]}),
                "regime_probabilities": pd.DataFrame({"regime_prob_0": [0.9]}),
                "risk_metrics": pd.DataFrame({"regime_id": [0], "historical_var_95": [0.12]}),
                "scenario_results": pd.DataFrame({"regime_id": [0], "mean_return": [0.01]}),
            }
            for name, frame in cached_payload.items():
                frame.to_parquet(temp_path / f"{name}.parquet")

            with patch("quantrisk.pipeline.get_data_dir", return_value=temp_path), patch.object(
                DataIngestion,
                "save",
                side_effect=RuntimeError("live failure"),
            ):
                results = run_pipeline(
                    start_date="2020-01-01",
                    end_date="2021-01-01",
                    portfolio_weights={"sp500": 1.0},
                    n_regimes=2,
                )
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)

        self.assertEqual(set(results), set(cached_payload))
        self.assertEqual(float(results["risk_metrics"].iloc[0]["historical_var_95"]), 0.12)

    def test_pipeline_normalizes_weights_and_validates_dates(self) -> None:
        """Pipeline helpers should normalize weights and reject inverted date windows."""
        normalized = normalize_portfolio_weights({"sp500": 2.0, "gold": 1.0})
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertAlmostEqual(normalized["sp500"], 2.0 / 3.0)

        with self.assertRaises(ValueError):
            normalize_portfolio_weights({})

        with self.assertRaises(ValueError):
            validate_date_window("2024-01-02", "2024-01-01")

    def test_ingestion_validates_required_columns(self) -> None:
        """Ingestion should fail fast if required market or macro series are missing."""
        ingestion = DataIngestion(start_date="2020-01-01", end_date="2020-02-01", fred_api_key="test")
        incomplete = pd.DataFrame(
            {
                "sp500_close": [1.0, 2.0],
                "us10y_yield": [4.0, 4.1],
            },
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        with self.assertRaises(ValueError):
            ingestion.validate_dataset_columns(incomplete)

    def test_dashboard_chart_helpers_behave_consistently(self) -> None:
        """Chart helpers should provide stable color, zoom, and duration behavior."""
        self.assertEqual(hex_to_rgba("#112233", 0.5), "rgba(17, 34, 51, 0.5)")

        dates = pd.date_range("2024-01-01", periods=400, freq="D")
        figure = go.Figure().add_trace(go.Scatter(x=dates, y=np.arange(len(dates))))
        zoomed = apply_zoom_range(figure, pd.Series(np.arange(len(dates)), index=dates), "1Y")
        x_range = zoomed.layout.xaxis.range
        self.assertIsNotNone(x_range)

        labels = pd.DataFrame(
            {"regime_name": ["Bull", "Bull", "Bear", "Bear", "Bull"]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        duration = build_regime_duration_table(labels)
        self.assertIn("avg_days", duration.columns)
        self.assertIn("Bull", duration.index)


if __name__ == "__main__":
    unittest.main()
