"""Public package API for Quantrisk."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DataIngestion",
    "FeatureEngineer",
    "RegimeDetector",
    "RiskModeler",
    "ScenarioEngine",
    "Backtester",
    "run_pipeline",
]

_EXPORTS = {
    "DataIngestion": "quantrisk.ingestion",
    "FeatureEngineer": "quantrisk.features",
    "RegimeDetector": "quantrisk.regime",
    "RiskModeler": "quantrisk.risk",
    "ScenarioEngine": "quantrisk.scenario",
    "Backtester": "quantrisk.backtest",
    "run_pipeline": "quantrisk.pipeline",
}


def __getattr__(name: str) -> object:
    """Lazily resolve public exports so optional dependencies load on demand."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'quantrisk' has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    return getattr(module, name)
