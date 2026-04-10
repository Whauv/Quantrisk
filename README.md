# Quantrisk

Quantrisk is a regime-aware portfolio analytics and risk dashboard built on free market and macroeconomic data. The project combines data ingestion, feature engineering, hidden Markov regime detection, risk modeling, scenario analysis, and backtesting into one local Python package with a Streamlit interface.

## Overview

Quantrisk is designed to answer a practical portfolio question:

How does portfolio risk change when the market shifts between bull, bear, crisis, and low-volatility regimes?

The pipeline:

- pulls market data from Yahoo Finance via `yfinance`
- pulls macro data from FRED via `fredapi`
- builds rolling volatility, momentum, yield-curve, and cross-asset correlation features
- fits a Gaussian Hidden Markov Model to detect market regimes
- computes regime-conditional risk metrics such as covariance, VaR, and CVaR
- runs scenario analysis and Monte Carlo stress testing
- backtests a regime-aware allocation strategy against a static benchmark
- visualizes the full workflow in a Streamlit dashboard

## Repository Layout

```text
quantrisk/
|-- .github/
|-- data/
|-- notebooks/
|-- src/
|   `-- quantrisk/
|       |-- __init__.py
|       |-- pipeline.py
|       |-- ingestion.py
|       |-- features.py
|       |-- regime.py
|       |-- risk.py
|       |-- scenario.py
|       |-- backtest.py
|       |-- alpha.py
|       |-- pricing.py
|       |-- sentiment.py
|       `-- dashboard/
|           |-- __init__.py
|           |-- app.py
|           |-- charting.py
|           |-- resources.py
|           |-- styling.py
|           `-- assets/
|-- tests/
|-- .env.example
|-- AGENTS.md
|-- CONTRIBUTING.md
|-- LICENSE
|-- README.md
|-- requirements.txt
`-- setup.py
```

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

## Environment Variables

Quantrisk uses free data sources only, but FRED requires an API key.

```powershell
$env:FRED_API_KEY="your_fred_api_key"
```

To persist it for future shells on Windows:

```powershell
setx FRED_API_KEY "your_fred_api_key"
```

See [.env.example](c:/Users/prana/OneDrive/Documents/Playground/quantrisk/.env.example) for the tracked template.

## Running the Dashboard

From the repository root:

```powershell
python -m streamlit run src\quantrisk\dashboard\app.py
```

The dashboard typically runs at [http://localhost:8501](http://localhost:8501).

## Running the Pipeline in Python

```python
from quantrisk import run_pipeline

results = run_pipeline(
    start_date="2010-01-01",
    end_date="2026-04-10",
    portfolio_weights={
        "sp500": 0.35,
        "msci_world": 0.20,
        "vix": 0.05,
        "gold": 0.15,
        "oil": 0.10,
        "eurusd": 0.15,
    },
    n_regimes=4,
)

feature_matrix = results["feature_matrix"]
risk_metrics = results["risk_metrics"]
```

## Testing

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

## Notes

- Data sourcing relies on free-tier providers, so provider outages or schema changes may affect live runs.
- Cached Parquet artifacts in `data/` allow the dashboard to reuse previous successful pipeline runs.
- `alpha.py`, `pricing.py`, and `sentiment.py` are placeholder modules reserved for future work.

## Contributing

See [CONTRIBUTING.md](c:/Users/prana/OneDrive/Documents/Playground/quantrisk/CONTRIBUTING.md) for workflow and contribution guidance.
