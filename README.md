# Quantrisk

Quantrisk is a regime-aware portfolio analytics and risk dashboard built on free market and macroeconomic data. It combines data ingestion, feature engineering, hidden Markov regime detection, risk modeling, scenario analysis, and backtesting into one local Python package with a Streamlit interface.

## What the project does

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

## Core modules

### Data layer

- `DataIngestion`
  Fetches daily OHLCV market data for S&P 500, MSCI World proxy, VIX, Gold, Oil, and EUR/USD from Yahoo Finance, plus Treasury and credit-spread series from FRED.

### Feature layer

- `FeatureEngineer`
  Computes rolling volatility, 1M or 3M or 6M momentum, yield-curve slope, rolling cross-asset correlations, and rolling z-score normalization.

### Regime layer

- `RegimeDetector`
  Fits a Gaussian HMM using `hmmlearn`, assigns regime probabilities, and auto-labels states such as `Bull`, `Bear`, `High-Vol Crisis`, and `Low-Vol Grind`.

### Risk layer

- `RiskModeler`
  Builds regime-conditional Ledoit-Wolf covariance estimates and computes historical and parametric VaR and CVaR across regimes.

### Scenario layer

- `ScenarioEngine`
  Supports custom shocks, Monte Carlo scenario analysis, and historical replay workflows.

### Strategy layer

- `Backtester`
  Evaluates a regime-aware allocation strategy against a static 60/40 benchmark using Yahoo Finance proxy assets.

### Presentation layer

- `dashboard/app.py`
  Streamlit dashboard for regime timelines, risk analysis, scenario stress testing, and backtest visualization.

## Dashboard pages

The Streamlit app currently includes:

- `Regime Timeline`
  S&P 500 price with regime-aware context and regime probability views.
- `Risk Metrics`
  Regime-specific VaR, CVaR, and correlation diagnostics.
- `Scenario Stress Test`
  Custom shock inputs, Monte Carlo distribution analysis, and historical scenario replay.
- `Backtesting`
  Regime-aware strategy versus a static benchmark with performance metrics and cumulative return charts.

## Project structure

```text
quantrisk/
|-- dashboard/
|   |-- app.py
|   `-- assets/
|-- data/
|-- notebooks/
|-- quantrisk/
|   |-- __init__.py
|   |-- ingestion.py
|   |-- features.py
|   |-- regime.py
|   |-- risk.py
|   |-- scenario.py
|   |-- backtest.py
|   |-- alpha.py
|   |-- pricing.py
|   `-- sentiment.py
|-- tests/
|   `-- test_all.py
|-- requirements.txt
|-- setup.py
`-- README.md
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
```

### 3. Install the package locally

```powershell
pip install -e .
```

## Environment variables

Quantrisk uses free data sources only, but FRED requires an API key.

Set your FRED key in PowerShell:

```powershell
$env:FRED_API_KEY="your_fred_api_key"
```

To persist it for future shells on Windows:

```powershell
setx FRED_API_KEY "your_fred_api_key"
```

## Run the dashboard

From the repository root:

```powershell
python -m streamlit run dashboard\app.py
```

The dashboard typically runs at:

- [http://localhost:8501](http://localhost:8501)

## Run the pipeline in Python

```python
from quantrisk import run_pipeline

results = run_pipeline(
    start_date="2010-01-01",
    end_date="2026-03-13",
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

## Saved artifacts

Pipeline outputs are written locally to the `data/` directory as Parquet files. These cached artifacts allow the dashboard to reuse previously computed results when live provider calls fail or are temporarily unavailable.

Typical artifacts include:

- `ingested_data.parquet`
- `feature_matrix.parquet`
- `returns.parquet`
- `regime_labels.parquet`
- `regime_probabilities.parquet`
- `risk_metrics.parquet`
- `scenario_results.parquet`
- `main_with_regimes.parquet`

## Testing

Run the basic test suite with:

```powershell
pytest
```

## Notes and limitations

- Data sourcing relies on free-tier providers, so provider outages or schema changes may affect live runs.
- FRED access requires a valid API key.
- Yahoo Finance responses can vary in column shape, especially when downloading multiple tickers.
- The Streamlit dashboard is under active UI iteration and should be treated as an evolving front end rather than a finalized product.

## Roadmap ideas

- stronger dashboard navigation and layout stabilization
- richer export workflows for reports and figures
- more explicit cache-status and data-source health messaging
- expanded unit coverage for feature engineering, regime detection, and backtesting
- optional deployment packaging for Streamlit Community Cloud or containerized hosting

## Publishing to GitHub

Suggested steps after final review:

```powershell
git init
git add .
git commit -m "Initial Quantrisk project"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

If the repository already exists, just skip `git init` and `git branch -M main`.

## License

No license file is included yet. If you plan to publish this publicly, add a license before sharing the repository broadly.
"# Quantrisk" 
