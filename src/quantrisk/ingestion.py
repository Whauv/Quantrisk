"""Data ingestion utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataIngestion:
    """Fetch, align, and persist market and macro data for the project."""

    start_date: str
    end_date: str
    fred_api_key: str | None = None
    output_path: Path | str = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "market_macro_data.parquet"
    )

    market_symbols: dict[str, str] = field(
        default_factory=lambda: {
            "sp500": "^GSPC",
            "msci_world": "URTH",
            "vix": "^VIX",
            "gold": "GC=F",
            "oil": "CL=F",
            "eurusd": "EURUSD=X",
        }
    )
    fred_series: dict[str, str] = field(
        default_factory=lambda: {
            "us10y_yield": "DGS10",
            "us2y_yield": "DGS2",
            "hy_credit_spread": "BAMLH0A0HYM2",
            "ig_credit_spread": "BAMLC0A0CM",
        }
    )

    def __post_init__(self) -> None:
        """Normalize configured paths and credentials after initialization."""
        self.output_path = Path(self.output_path)
        self.validate_dates()
        if self.fred_api_key is None:
            self.fred_api_key = os.getenv("FRED_API_KEY")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.output_path.parent / "yfinance_cache"))

    def validate_dates(self) -> None:
        """Validate that the requested window is chronologically correct."""
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        if start >= end:
            raise ValueError("`start_date` must be earlier than `end_date`.")

    def normalize_yfinance_columns(self, data: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """Flatten Yahoo Finance columns into a stable single-level schema."""
        flattened_columns: list[str] = []

        for column in data.columns:
            if isinstance(column, tuple):
                column_name = str(column[0]).lower()
            else:
                column_name = str(column).lower()
            flattened_columns.append(f"{asset_name}_{column_name}")

        normalized = data.copy()
        normalized.columns = flattened_columns
        return normalized

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch daily OHLCV data for configured market symbols from Yahoo Finance."""
        frames: list[pd.DataFrame] = []
        failures: list[str] = []

        for asset_name, symbol in self.market_symbols.items():
            try:
                data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
            except Exception as error:
                failures.append(f"{asset_name} ({symbol}): {error}")
                continue
            if data.empty:
                failures.append(f"{asset_name} ({symbol}): empty response")
                continue

            normalized = self.normalize_yfinance_columns(data, asset_name)
            normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
            frames.append(normalized)

        if failures:
            LOGGER.warning("Yahoo Finance fetch issues: %s", "; ".join(failures))

        if not frames:
            return pd.DataFrame()

        market_data = pd.concat(frames, axis=1).sort_index()
        market_data.index.name = "date"
        return market_data

    def fetch_macro_data(self) -> pd.DataFrame:
        """Fetch configured macroeconomic time series from the FRED API."""
        if not self.fred_api_key:
            raise ValueError("A FRED API key is required. Set `fred_api_key` or the `FRED_API_KEY` environment variable.")

        fred = Fred(api_key=self.fred_api_key)
        series_frames: list[pd.Series] = []
        failures: list[str] = []

        for column_name, series_id in self.fred_series.items():
            try:
                series = fred.get_series(series_id, observation_start=self.start_date, observation_end=self.end_date)
            except Exception as error:
                failures.append(f"{column_name} ({series_id}): {error}")
                continue
            if series.empty:
                failures.append(f"{column_name} ({series_id}): empty response")
                continue

            normalized = pd.Series(series, name=column_name)
            normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
            series_frames.append(normalized)

        if failures:
            LOGGER.warning("FRED fetch issues: %s", "; ".join(failures))

        if not series_frames:
            return pd.DataFrame()

        macro_data = pd.concat(series_frames, axis=1).sort_index()
        macro_data.index.name = "date"
        return macro_data

    def validate_dataset_columns(self, dataset: pd.DataFrame) -> None:
        """Validate that required market and macro columns are present after ingestion."""
        missing_market_assets = [
            asset_name
            for asset_name in self.market_symbols
            if not any(
                column in dataset.columns
                for column in (f"{asset_name}_adj close", f"{asset_name}_close")
            )
        ]
        missing_macro_series = [series_name for series_name in self.fred_series if series_name not in dataset.columns]

        if missing_market_assets or missing_macro_series:
            message_parts: list[str] = []
            if missing_market_assets:
                message_parts.append(f"market assets: {missing_market_assets}")
            if missing_macro_series:
                message_parts.append(f"macro series: {missing_macro_series}")
            raise ValueError("Ingestion did not return all required data sources. Missing " + "; ".join(message_parts))

    def build_dataset(self) -> pd.DataFrame:
        """Fetch all sources, align them to a shared daily index, and forward-fill gaps."""
        market_data = self.fetch_market_data()
        macro_data = self.fetch_macro_data()

        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        daily_index = pd.date_range(start=start, end=end, freq="D", name="date")

        dataset = pd.concat([market_data, macro_data], axis=1).reindex(daily_index).sort_index().ffill()
        if dataset.empty or dataset.dropna(how="all").empty:
            raise ValueError("No live data could be fetched from Yahoo Finance or FRED for the selected window.")
        self.validate_dataset_columns(dataset)
        return dataset

    def save(self) -> pd.DataFrame:
        """Build the combined dataset and save it as a Parquet file."""
        dataset = self.build_dataset()
        dataset.to_parquet(self.output_path)
        return dataset

    def load(self) -> pd.DataFrame:
        """Load the persisted Parquet dataset from disk."""
        return pd.read_parquet(self.output_path)
