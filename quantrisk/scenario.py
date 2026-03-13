"""Scenario analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScenarioEngine:
    """Summarize portfolio behavior under historically observed market regimes."""

    returns: pd.DataFrame
    regime_data: pd.DataFrame | pd.Series
    portfolio_weights: dict[str, float]

    def __post_init__(self) -> None:
        """Normalize inputs and validate that portfolio assets are available."""
        if self.returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        self.returns = self.returns.sort_index().copy()
        self.returns.index = pd.to_datetime(self.returns.index)

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

    def get_portfolio_returns(self) -> pd.Series:
        """Compute daily portfolio returns from asset returns and portfolio weights."""
        weights = pd.Series(self.portfolio_weights, dtype=float).reindex(self.returns.columns, fill_value=0.0)
        portfolio_returns = self.returns.mul(weights, axis=1).sum(axis=1)
        return portfolio_returns.rename("portfolio_return")

    def apply_shocks(self, shocks: dict[str, float]) -> pd.Series:
        """Apply one-period asset shocks to the configured portfolio and return shocked P&L."""
        weights = pd.Series(self.portfolio_weights, dtype=float).reindex(self.returns.columns, fill_value=0.0)
        shock_vector = pd.Series(shocks, dtype=float).reindex(self.returns.columns, fill_value=0.0)
        shocked_pnl = float((weights * shock_vector).sum())
        return pd.Series({"shocked_portfolio_return": shocked_pnl})

    def run_monte_carlo(
        self,
        shocks: dict[str, float] | None = None,
        n_simulations: int = 5000,
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate a one-day portfolio P&L distribution using empirical mean and covariance."""
        weights = pd.Series(self.portfolio_weights, dtype=float).reindex(self.returns.columns, fill_value=0.0)
        adjusted_returns = self.returns.copy()

        if shocks:
            shock_vector = pd.Series(shocks, dtype=float).reindex(adjusted_returns.columns, fill_value=0.0)
            adjusted_returns = adjusted_returns.add(shock_vector, axis=1)

        mean_vector = adjusted_returns.mean().to_numpy()
        covariance_matrix = adjusted_returns.cov().to_numpy()
        rng = np.random.default_rng(seed=random_state)
        simulated_asset_returns = rng.multivariate_normal(mean_vector, covariance_matrix, size=n_simulations)
        portfolio_pnl = simulated_asset_returns @ weights.to_numpy()

        pnl_frame = pd.DataFrame({"portfolio_pnl": portfolio_pnl})
        metric_rows: list[dict[str, float]] = []

        for confidence_level in confidence_levels:
            cutoff = float(np.quantile(portfolio_pnl, 1.0 - confidence_level))
            tail = portfolio_pnl[portfolio_pnl <= cutoff]
            metric_rows.append(
                {
                    "confidence_level": confidence_level,
                    "var": float(-cutoff),
                    "cvar": float(-tail.mean()) if len(tail) > 0 else float("nan"),
                }
            )

        metrics = pd.DataFrame(metric_rows)
        return pnl_frame, metrics

    def historical_replay(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return portfolio performance over a selected historical window."""
        portfolio_returns = self.get_portfolio_returns()
        replay = portfolio_returns.loc[start_date:end_date].to_frame()
        if replay.empty:
            raise ValueError("No returns are available for the selected historical replay window.")

        replay["cumulative_return"] = (1.0 + replay["portfolio_return"]).cumprod() - 1.0
        return replay

    def run(self) -> pd.DataFrame:
        """Return a regime-conditioned scenario summary for the configured portfolio."""
        portfolio_returns = self.get_portfolio_returns()
        scenario_frame = portfolio_returns.to_frame().join(self.regime_data, how="inner").dropna()

        if scenario_frame.empty:
            raise ValueError("No overlapping dates were found between returns and regime labels.")

        summaries: list[dict[str, float | int | str]] = []

        for regime_id, group in scenario_frame.groupby("regime_id"):
            summary: dict[str, float | int | str] = {
                "regime_id": int(regime_id),
                "observations": int(len(group)),
                "mean_return": float(group["portfolio_return"].mean()),
                "volatility": float(group["portfolio_return"].std(ddof=0)),
                "best_day": float(group["portfolio_return"].max()),
                "worst_day": float(group["portfolio_return"].min()),
                "cumulative_return": float((1.0 + group["portfolio_return"]).prod() - 1.0),
            }
            if "regime_name" in group.columns:
                summary["regime_name"] = str(group["regime_name"].mode().iloc[0])
            summaries.append(summary)

        return pd.DataFrame(summaries).sort_values("regime_id").reset_index(drop=True)
