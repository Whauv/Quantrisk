"""Market regime detection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(slots=True)
class RegimeDetector:
    """Fit a Gaussian HMM on engineered features and annotate market regimes."""

    features: pd.DataFrame
    main_data: pd.DataFrame | None = None
    n_regimes: int = 4
    covariance_type: str = "diag"
    n_iter: int = 500
    random_state: int = 42
    model: GaussianHMM | None = field(default=None, init=False)
    regime_stats_: pd.DataFrame | None = field(default=None, init=False)
    regime_names_: dict[int, str] = field(default_factory=dict, init=False)
    regime_frame_: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate detector configuration and normalize the feature index."""
        if self.n_regimes < 2:
            raise ValueError("`n_regimes` must be at least 2.")
        if self.features.empty:
            raise ValueError("Feature matrix is empty. Fit cannot proceed.")
        normalized = self.features.copy()
        normalized.index = pd.to_datetime(normalized.index)
        if normalized.isna().any().any():
            raise ValueError("Feature matrix must not contain NaN values before fitting the HMM.")
        self.features = normalized.sort_index()

    def fit(self) -> pd.DataFrame:
        """Fit the HMM, infer regime labels and probabilities, and cache the results."""
        if len(self.features) < self.n_regimes * 20:
            raise ValueError("Feature matrix is too short relative to the requested number of regimes.")

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        feature_values = self.features.to_numpy()
        self.model.fit(feature_values)

        regime_ids = self.model.predict(feature_values)
        probabilities = self.model.predict_proba(feature_values)

        regime_frame = pd.DataFrame(index=self.features.index)
        regime_frame["regime_id"] = regime_ids

        for regime_id in range(self.n_regimes):
            regime_frame[f"regime_prob_{regime_id}"] = probabilities[:, regime_id]

        regime_names = self.name_regimes(regime_frame["regime_id"])
        regime_frame["regime_name"] = regime_frame["regime_id"].map(regime_names)

        self.regime_frame_ = regime_frame
        return regime_frame

    def name_regimes(self, regime_ids: pd.Series) -> dict[int, str]:
        """Assign descriptive labels to regimes using mean volatility and return proxies."""
        volatility_columns = [column for column in self.features.columns if "_vol_" in column]
        return_columns = [column for column in self.features.columns if "_mom_1m" in column]

        if not volatility_columns or not return_columns:
            raise ValueError("Feature matrix must contain volatility and 1-month momentum columns to name regimes.")

        stats = pd.DataFrame(index=sorted(regime_ids.unique()))
        stats["mean_volatility"] = regime_ids.to_frame("regime_id").join(self.features[volatility_columns]).groupby("regime_id").mean().mean(axis=1)
        stats["mean_return"] = regime_ids.to_frame("regime_id").join(self.features[return_columns]).groupby("regime_id").mean().mean(axis=1)

        assigned_names: dict[int, str] = {}
        available_regimes = list(stats.index)

        crisis_regime = stats["mean_volatility"].idxmax()
        assigned_names[crisis_regime] = "High-Vol Crisis"
        available_regimes.remove(crisis_regime)

        if available_regimes:
            grind_regime = stats.loc[available_regimes, "mean_volatility"].idxmin()
            assigned_names[grind_regime] = "Low-Vol Grind"
            available_regimes.remove(grind_regime)

        if available_regimes:
            bull_regime = stats.loc[available_regimes, "mean_return"].idxmax()
            assigned_names[bull_regime] = "Bull"
            available_regimes.remove(bull_regime)

        if available_regimes:
            bear_regime = stats.loc[available_regimes, "mean_return"].idxmin()
            assigned_names[bear_regime] = "Bear"
            available_regimes.remove(bear_regime)

        for regime_id in available_regimes:
            assigned_names[regime_id] = f"Regime {regime_id}"

        self.regime_stats_ = stats.sort_index()
        self.regime_names_ = assigned_names
        return assigned_names

    def validate(self) -> pd.DataFrame:
        """Print and return regime statistics for selected historical stress windows."""
        if self.regime_frame_ is None:
            raise ValueError("Run `fit()` before validation.")

        periods = {
            "GFC": ("2008-09-01", "2009-03-31"),
            "COVID": ("2020-02-01", "2020-04-30"),
            "Rate Hike cycle": ("2022-01-01", "2022-12-31"),
        }

        summaries: list[pd.DataFrame] = []

        for period_name, (start_date, end_date) in periods.items():
            period_frame = self.regime_frame_.loc[start_date:end_date]
            if period_frame.empty:
                continue

            probability_columns = [column for column in period_frame.columns if column.startswith("regime_prob_")]
            summary = pd.DataFrame(
                {
                    "period": period_name,
                    "start_date": pd.Timestamp(start_date),
                    "end_date": pd.Timestamp(end_date),
                    "observations": len(period_frame),
                    "dominant_regime_id": int(period_frame["regime_id"].mode().iloc[0]),
                    "dominant_regime_name": period_frame["regime_name"].mode().iloc[0],
                    "mean_max_probability": period_frame[probability_columns].max(axis=1).mean(),
                },
                index=[0],
            )
            summaries.append(summary)

        validation = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

        if not validation.empty:
            print(validation.to_string(index=False))
        else:
            print("No validation periods overlap the fitted regime sample.")

        return validation

    def save_to_main_dataframe(self, main_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Join regime labels and probabilities back onto the main date-indexed dataset."""
        if self.regime_frame_ is None:
            raise ValueError("Run `fit()` before saving regime outputs.")

        target = main_data if main_data is not None else self.main_data
        if target is None:
            raise ValueError("A main DataFrame is required to save regime outputs.")

        combined = target.join(self.regime_frame_, how="left")
        self.main_data = combined
        return combined
