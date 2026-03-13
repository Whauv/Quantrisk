"""Risk modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf


@dataclass(slots=True)
class RiskModeler:
    """Estimate regime-aware covariance and portfolio tail-risk metrics."""

    returns: pd.DataFrame
    regime_data: pd.DataFrame | pd.Series
    portfolio_weights: dict[str, float]
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    covariance_matrices_: dict[int, pd.DataFrame] = field(default_factory=dict, init=False)
    risk_summary_: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Normalize inputs and validate the portfolio definition."""
        if self.returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        aligned_returns = self.returns.sort_index().copy()
        aligned_returns.index = pd.to_datetime(aligned_returns.index)
        self.returns = aligned_returns

        if isinstance(self.regime_data, pd.Series):
            regime_frame = self.regime_data.to_frame(name="regime_id")
        else:
            regime_frame = self.regime_data.copy()

        if "regime_id" not in regime_frame.columns:
            raise KeyError("Regime data must contain a `regime_id` column.")

        regime_frame.index = pd.to_datetime(regime_frame.index)
        self.regime_data = regime_frame.sort_index()

        missing_assets = sorted(set(self.portfolio_weights) - set(self.returns.columns))
        if missing_assets:
            raise KeyError(f"Portfolio weights reference assets not present in returns: {missing_assets}")

    def get_aligned_data(self) -> pd.DataFrame:
        """Align asset returns with regime labels on a shared date index."""
        regime_columns = ["regime_id"]
        if "regime_name" in self.regime_data.columns:
            regime_columns.append("regime_name")

        aligned = self.returns.join(self.regime_data[regime_columns], how="inner").dropna()
        if aligned.empty:
            raise ValueError("No overlapping dates were found between returns and regime labels.")
        return aligned

    def get_portfolio_weight_vector(self) -> pd.Series:
        """Return portfolio weights as a Series aligned to the returns columns."""
        weights = pd.Series(self.portfolio_weights, dtype=float)
        weights = weights.reindex(self.returns.columns, fill_value=0.0)
        return weights

    def compute_regime_covariances(self) -> dict[int, pd.DataFrame]:
        """Estimate a Ledoit-Wolf covariance matrix for each observed regime."""
        aligned = self.get_aligned_data()
        covariance_matrices: dict[int, pd.DataFrame] = {}

        for regime_id, regime_frame in aligned.groupby("regime_id"):
            regime_returns = regime_frame[self.returns.columns].dropna()
            if len(regime_returns) < 2:
                continue

            estimator = LedoitWolf()
            estimator.fit(regime_returns.to_numpy())
            covariance = pd.DataFrame(
                estimator.covariance_,
                index=self.returns.columns,
                columns=self.returns.columns,
            )
            covariance_matrices[int(regime_id)] = covariance

        self.covariance_matrices_ = covariance_matrices
        return covariance_matrices

    def compute_historical_var(self, portfolio_returns: pd.Series, confidence_level: float) -> float:
        """Compute historical-simulation VaR as a positive portfolio loss threshold."""
        return float(-np.quantile(portfolio_returns.dropna(), 1.0 - confidence_level))

    def compute_parametric_var(
        self,
        portfolio_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        weights: pd.Series,
        confidence_level: float,
    ) -> float:
        """Compute variance-covariance VaR using regime-specific mean and covariance."""
        mean_return = float(portfolio_returns.mean())
        portfolio_volatility = float(np.sqrt(weights.to_numpy() @ covariance_matrix.to_numpy() @ weights.to_numpy()))
        z_score = float(norm.ppf(1.0 - confidence_level))
        return float(-(mean_return + z_score * portfolio_volatility))

    def compute_expected_shortfall(self, portfolio_returns: pd.Series, confidence_level: float) -> float:
        """Compute historical expected shortfall as the mean loss beyond the VaR cutoff."""
        cutoff = float(np.quantile(portfolio_returns.dropna(), 1.0 - confidence_level))
        tail_losses = portfolio_returns[portfolio_returns <= cutoff]
        if tail_losses.empty:
            return float("nan")
        return float(-tail_losses.mean())

    def compute_regime_risk_metrics(self) -> pd.DataFrame:
        """Compute covariance, VaR, ES, and correlation statistics for each regime."""
        aligned = self.get_aligned_data()
        weights = self.get_portfolio_weight_vector()
        covariance_matrices = self.compute_regime_covariances()

        summaries: list[dict[str, float | int]] = []

        for regime_id, regime_frame in aligned.groupby("regime_id"):
            regime_id_int = int(regime_id)
            if regime_id_int not in covariance_matrices:
                continue

            asset_returns = regime_frame[self.returns.columns].dropna()
            if asset_returns.empty:
                continue

            portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1)
            covariance = covariance_matrices[regime_id_int]
            correlation = asset_returns.corr()

            summary: dict[str, float | int] = {
                "regime_id": regime_id_int,
                "observations": int(len(asset_returns)),
                "portfolio_mean_return": float(portfolio_returns.mean()),
                "portfolio_volatility": float(portfolio_returns.std(ddof=0)),
                "average_correlation": float(correlation.where(~np.eye(len(correlation), dtype=bool)).stack().mean()),
            }
            if "regime_name" in regime_frame.columns:
                summary["regime_name"] = regime_frame["regime_name"].mode().iloc[0]

            for confidence_level in self.confidence_levels:
                level_suffix = str(int(confidence_level * 100))
                summary[f"historical_var_{level_suffix}"] = self.compute_historical_var(portfolio_returns, confidence_level)
                summary[f"parametric_var_{level_suffix}"] = self.compute_parametric_var(
                    portfolio_returns=portfolio_returns,
                    covariance_matrix=covariance,
                    weights=weights,
                    confidence_level=confidence_level,
                )
                summary[f"expected_shortfall_{level_suffix}"] = self.compute_expected_shortfall(
                    portfolio_returns,
                    confidence_level,
                )

            summaries.append(summary)

        risk_summary = pd.DataFrame(summaries)
        if risk_summary.empty:
            self.risk_summary_ = risk_summary
            return risk_summary

        risk_summary = risk_summary.sort_values("regime_id").reset_index(drop=True)
        self.risk_summary_ = risk_summary
        return risk_summary

    def compare_regimes(self) -> pd.DataFrame:
        """Return a regime-by-regime summary of correlation and tail-risk metrics."""
        if self.risk_summary_ is None:
            return self.compute_regime_risk_metrics()
        return self.risk_summary_.copy()
