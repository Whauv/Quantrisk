"""Feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureEngineer:
    """Generate normalized modeling features from the ingested market dataset."""

    data: pd.DataFrame
    volatility_windows: tuple[int, ...] = (20, 60, 90)
    momentum_windows: dict[str, int] = field(
        default_factory=lambda: {
            "1m": 21,
            "3m": 63,
            "6m": 126,
        }
    )
    correlation_window: int = 30
    standardization_window: int = 252

    def get_price_columns(self) -> dict[str, str]:
        """Map each asset name to its preferred price column in the dataset."""
        price_columns: dict[str, str] = {}

        for column in self.data.columns:
            if column.endswith("_adj close"):
                asset_name = column[: -len("_adj close")]
                price_columns[asset_name] = column
            elif column.endswith("_close") and column[: -len("_close")] not in price_columns:
                asset_name = column[: -len("_close")]
                price_columns[asset_name] = column

        return price_columns

    def compute_returns(self) -> pd.DataFrame:
        """Compute daily percentage returns for each asset price series."""
        price_columns = self.get_price_columns()
        prices = self.data.loc[:, list(price_columns.values())].copy()
        prices = prices.rename(columns={column: asset for asset, column in price_columns.items()})
        return prices.pct_change()

    def compute_rolling_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling annualized volatility features for each asset."""
        features: dict[str, pd.Series] = {}

        for asset in returns.columns:
            for window in self.volatility_windows:
                feature_name = f"{asset}_vol_{window}d"
                features[feature_name] = returns[asset].rolling(window=window).std() * np.sqrt(252.0)

        return pd.DataFrame(features, index=returns.index)

    def compute_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute trailing momentum features across configured horizons for each asset."""
        features: dict[str, pd.Series] = {}

        for asset in prices.columns:
            for label, window in self.momentum_windows.items():
                feature_name = f"{asset}_mom_{label}"
                features[feature_name] = prices[asset].pct_change(periods=window)

        return pd.DataFrame(features, index=prices.index)

    def compute_yield_curve_slope(self) -> pd.Series:
        """Compute the Treasury yield curve slope as 10Y minus 2Y yields."""
        slope = self.data["us10y_yield"] - self.data["us2y_yield"]
        return slope.rename("yield_curve_slope")

    def compute_cross_asset_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling correlations between the S&P 500 and all other assets."""
        if "sp500" not in returns.columns:
            raise KeyError("The dataset must include S&P 500 prices to compute cross-asset correlations.")

        features: dict[str, pd.Series] = {}
        benchmark = returns["sp500"]

        for asset in returns.columns:
            if asset == "sp500":
                continue
            feature_name = f"sp500_{asset}_corr_{self.correlation_window}d"
            features[feature_name] = benchmark.rolling(self.correlation_window).corr(returns[asset])

        return pd.DataFrame(features, index=returns.index)

    def standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score normalization using only prior observations."""
        rolling_mean = features.rolling(window=self.standardization_window).mean().shift(1)
        rolling_std = features.rolling(window=self.standardization_window).std(ddof=0).shift(1)
        standardized = (features - rolling_mean) / rolling_std.replace(0.0, np.nan)
        return standardized.replace([np.inf, -np.inf], np.nan)

    def build_feature_matrix(self) -> pd.DataFrame:
        """Build the full normalized feature matrix and drop rows with missing values."""
        price_columns = self.get_price_columns()
        prices = self.data.loc[:, list(price_columns.values())].copy()
        prices = prices.rename(columns={column: asset for asset, column in price_columns.items()})
        returns = prices.pct_change()

        volatility = self.compute_rolling_volatility(returns)
        momentum = self.compute_momentum(prices)
        slope = self.compute_yield_curve_slope().to_frame()
        correlation = self.compute_cross_asset_correlation(returns)

        raw_features = pd.concat([volatility, momentum, slope, correlation], axis=1).sort_index()
        standardized_features = self.standardize_features(raw_features)
        clean_features = standardized_features.dropna().copy()
        clean_features.index.name = self.data.index.name
        return clean_features
