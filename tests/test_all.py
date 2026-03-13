"""Basic smoke tests for the quantrisk package."""

from quantrisk import (
    DataIngestion,
    FeatureEngineer,
    RegimeDetector,
    RiskModeler,
    ScenarioEngine,
    __all__,
    run_pipeline,
)


def test_public_api_exports_exist() -> None:
    expected = {
        "DataIngestion",
        "FeatureEngineer",
        "RegimeDetector",
        "RiskModeler",
        "ScenarioEngine",
        "run_pipeline",
    }
    assert set(__all__) == expected
    assert callable(run_pipeline)
    assert DataIngestion.__name__ == "DataIngestion"
    assert FeatureEngineer.__name__ == "FeatureEngineer"
    assert RegimeDetector.__name__ == "RegimeDetector"
    assert RiskModeler.__name__ == "RiskModeler"
    assert ScenarioEngine.__name__ == "ScenarioEngine"
