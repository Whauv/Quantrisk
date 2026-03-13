"""Package configuration for local installation."""

from pathlib import Path

from setuptools import find_packages, setup


README_PATH = Path(__file__).with_name("README.md")


setup(
    name="quantrisk",
    version="0.1.0",
    description="Regime-aware portfolio analytics, risk modeling, and dashboard tooling.",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "yfinance",
        "pyarrow",
        "hmmlearn",
        "scikit-learn",
        "cvxpy",
        "streamlit",
        "plotly",
        "transformers",
        "torch",
        "fredapi",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "jupyterlab",
        "notebook",
        "pytest",
        "requests",
        "tqdm",
    ],
    keywords=[
        "quant finance",
        "risk management",
        "hidden markov model",
        "streamlit dashboard",
        "portfolio analytics",
        "backtesting",
    ],
)
