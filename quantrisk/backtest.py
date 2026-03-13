"""Backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


@dataclass(slots=True)
class Backtester:
    """Run a regime-aware allocation backtest against a static benchmark."""

    regime_data: pd.DataFrame | pd.Series
    asset_returns: pd.DataFrame | None = None
    transaction_cost_bps: float = 10.0
    proxy_symbols: dict[str, str] = field(
        default_factory=lambda: {
            "equities": "SPY",
            "bonds": "TLT",
            "gold": "GLD",
        }
    )

    def __post_init__(self) -> None:
        """Normalize regime inputs and ensure the backtest can be aligned by date."""
        if isinstance(self.regime_data, pd.Series):
            regime_frame = self.regime_data.to_frame(name="regime_id")
        else:
            regime_frame = self.regime_data.copy()

        if "regime_name" not in regime_frame.columns:
            raise KeyError("Regime data must contain a `regime_name` column for strategy allocation rules.")

        regime_frame.index = pd.to_datetime(regime_frame.index)
        self.regime_data = regime_frame.sort_index()

        if self.asset_returns is not None:
            returns = self.asset_returns.copy()
            returns.index = pd.to_datetime(returns.index)
            self.asset_returns = returns.sort_index()

    def get_price_series(self, data: pd.DataFrame) -> pd.Series:
        """Extract the adjusted-close or close series from a Yahoo Finance download."""
        if isinstance(data.columns, pd.MultiIndex):
            if ("Adj Close", data.columns[0][1]) in data.columns:
                return data[("Adj Close", data.columns[0][1])]
            if ("Close", data.columns[0][1]) in data.columns:
                return data[("Close", data.columns[0][1])]

        for column_name in ("Adj Close", "Close"):
            if column_name in data.columns:
                return data[column_name]

        first_column = data.columns[0]
        return data[first_column]

    def fetch_proxy_returns(self) -> pd.DataFrame:
        """Fetch daily adjusted-close returns for SPY, TLT, and GLD from Yahoo Finance."""
        start_date = self.regime_data.index.min().strftime("%Y-%m-%d")
        end_date = (self.regime_data.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        frames: list[pd.Series] = []

        for asset_name, symbol in self.proxy_symbols.items():
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if data.empty:
                raise ValueError(f"No Yahoo Finance data returned for proxy {symbol}.")

            returns = self.get_price_series(data).pct_change().rename(asset_name)
            returns.index = pd.to_datetime(returns.index).tz_localize(None)
            frames.append(returns)

        proxy_returns = pd.concat(frames, axis=1).dropna().sort_index()
        return proxy_returns

    def get_asset_returns(self) -> pd.DataFrame:
        """Return aligned proxy returns, fetching them from Yahoo Finance when needed."""
        if self.asset_returns is None:
            self.asset_returns = self.fetch_proxy_returns()

        required_assets = set(self.proxy_symbols)
        missing_assets = sorted(required_assets - set(self.asset_returns.columns))
        if missing_assets:
            raise KeyError(f"Asset returns must contain columns for proxy assets: {missing_assets}")

        return self.asset_returns.loc[:, list(self.proxy_symbols.keys())].sort_index()

    def get_target_weights(self, regime_name: str) -> dict[str, float]:
        """Map a regime label to the strategy's target portfolio weights."""
        if regime_name == "Bull":
            return {"equities": 0.8, "bonds": 0.2, "gold": 0.0}
        if regime_name in {"Bear", "High-Vol Crisis"}:
            return {"equities": 0.3, "bonds": 0.7, "gold": 0.0}
        if regime_name == "Low-Vol Grind":
            return {"equities": 1.0 / 3.0, "bonds": 1.0 / 3.0, "gold": 1.0 / 3.0}
        return {"equities": 0.6, "bonds": 0.4, "gold": 0.0}

    def build_monthly_weights(self, aligned: pd.DataFrame, benchmark: bool = False) -> pd.DataFrame:
        """Create monthly target weights for the strategy or the static benchmark."""
        rebalancing_dates = aligned.groupby(aligned.index.to_period("M")).head(1).index
        weight_rows: list[pd.Series] = []

        for date in rebalancing_dates:
            if benchmark:
                target = {"equities": 0.6, "bonds": 0.4, "gold": 0.0}
            else:
                regime_name = str(aligned.loc[date, "regime_name"])
                target = self.get_target_weights(regime_name)
            weight_rows.append(pd.Series(target, name=date))

        weights = pd.DataFrame(weight_rows).sort_index()
        weights = weights.reindex(aligned.index).ffill()
        return weights

    def apply_transaction_costs(self, weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Compute per-day transaction costs and monthly turnover from weight changes."""
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum(axis=0))
        cost_rate = self.transaction_cost_bps / 10000.0
        costs = turnover * cost_rate
        monthly_turnover = turnover.resample("ME").sum()
        return costs, monthly_turnover

    def compute_portfolio_path(self, benchmark: bool = False) -> pd.DataFrame:
        """Compute daily net returns and cumulative performance for one portfolio."""
        asset_returns = self.get_asset_returns()
        aligned = asset_returns.join(self.regime_data[["regime_name"]], how="inner").dropna()
        if aligned.empty:
            raise ValueError("No overlapping dates were found between proxy returns and regime labels.")

        weights = self.build_monthly_weights(aligned, benchmark=benchmark)
        asset_only = aligned.loc[:, list(self.proxy_symbols.keys())]
        gross_returns = (asset_only * weights).sum(axis=1)
        costs, monthly_turnover = self.apply_transaction_costs(weights)
        net_returns = gross_returns - costs
        cumulative_returns = (1.0 + net_returns).cumprod()

        result = pd.DataFrame(
            {
                "gross_return": gross_returns,
                "transaction_cost": costs,
                "net_return": net_returns,
                "cumulative_return": cumulative_returns,
            },
            index=aligned.index,
        )
        result.attrs["monthly_turnover"] = monthly_turnover
        result.attrs["weights"] = weights
        return result

    def compute_metrics(self, portfolio_path: pd.DataFrame) -> dict[str, float]:
        """Compute standard backtest performance metrics from daily net returns."""
        returns = portfolio_path["net_return"]
        downside = returns[returns < 0.0]
        cumulative = portfolio_path["cumulative_return"]
        running_peak = cumulative.cummax()
        drawdown = cumulative / running_peak - 1.0

        annualized_return = float((1.0 + returns.mean()) ** 252 - 1.0)
        annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
        sharpe_ratio = float(annualized_return / annualized_volatility) if annualized_volatility > 0 else np.nan
        downside_volatility = float(downside.std(ddof=0) * np.sqrt(252.0)) if not downside.empty else np.nan
        sortino_ratio = float(annualized_return / downside_volatility) if downside_volatility and downside_volatility > 0 else np.nan
        maximum_drawdown = float(drawdown.min())
        monthly_turnover = float(portfolio_path.attrs["monthly_turnover"].mean())

        return {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "maximum_drawdown": maximum_drawdown,
            "monthly_turnover": monthly_turnover,
        }

    def build_cumulative_return_plot(
        self,
        strategy_path: pd.DataFrame,
        benchmark_path: pd.DataFrame,
    ) -> go.Figure:
        """Create a Plotly chart comparing cumulative returns."""
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=strategy_path.index,
                y=strategy_path["cumulative_return"],
                mode="lines",
                name="Regime-Aware Strategy",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=benchmark_path.index,
                y=benchmark_path["cumulative_return"],
                mode="lines",
                name="Static 60/40 Benchmark",
            )
        )
        figure.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template="plotly_white",
        )
        return figure

    def run_backtest(self) -> tuple[pd.DataFrame, go.Figure]:
        """Run the strategy and benchmark backtests and return metrics plus the plot."""
        strategy_path = self.compute_portfolio_path(benchmark=False)
        benchmark_path = self.compute_portfolio_path(benchmark=True)

        comparison = pd.DataFrame(
            {
                "Regime-Aware Strategy": self.compute_metrics(strategy_path),
                "Static 60/40 Benchmark": self.compute_metrics(benchmark_path),
            }
        ).T

        figure = self.build_cumulative_return_plot(strategy_path, benchmark_path)
        return comparison, figure
