"""Chart-building and visualization helpers for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.styling import is_light_theme

REGIME_COLORS = {
    "Bull": "#3a914b",
    "Bear": "#c43d3d",
    "High-Vol Crisis": "#8e24aa",
    "Low-Vol Grind": "#1f77b4",
}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba color string for Plotly."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, received: {hex_color}")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def get_sp500_price_column(frame: pd.DataFrame) -> str:
    """Return the preferred S&P 500 price column from the ingested dataset."""
    for column in ("sp500_adj close", "sp500_close"):
        if column in frame.columns:
            return column
    raise KeyError("S&P 500 price column was not found in the ingested dataset.")


def format_plotly(figure: go.Figure) -> go.Figure:
    """Apply a theme-aware visual style to Plotly figures."""
    light_theme = is_light_theme()
    figure.update_layout(
        template="plotly_white" if light_theme else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)" if light_theme else "rgba(255,255,255,0.02)",
        hovermode="x unified",
        font={"color": "#0f172a" if light_theme else "#f8fafc"},
        margin={"l": 48, "r": 28, "t": 70, "b": 48},
    )
    return figure


def apply_zoom_range(figure: go.Figure, frame: pd.DataFrame | pd.Series, zoom_label: str) -> go.Figure:
    """Apply a date-range zoom preset to a Plotly time-series figure."""
    if zoom_label == "Full":
        return figure

    index = frame.index
    if len(index) == 0:
        return figure

    end_date = pd.to_datetime(index.max())
    zoom_map = {"6M": 183, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
    start_date = end_date - pd.Timedelta(days=zoom_map[zoom_label])
    figure.update_xaxes(range=[start_date, end_date])
    return figure


def filter_artifacts_by_date(artifacts: dict[str, pd.DataFrame], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Filter date-indexed artifacts to a recent window for faster visuals."""
    filtered = artifacts.copy()
    end_date = pd.to_datetime(artifacts["main_data"].index.max())
    start_cutoff = end_date - pd.Timedelta(days=lookback_days)
    for key in ("main_data", "returns", "feature_matrix", "regime_labels", "regime_probabilities"):
        filtered[key] = filtered[key].loc[filtered[key].index >= start_cutoff].copy()
    return filtered


def build_regime_timeline_figure(price_frame: pd.DataFrame) -> go.Figure:
    """Build the S&P 500 timeline chart with regime shading."""
    sp500_column = get_sp500_price_column(price_frame)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=price_frame.index,
            y=price_frame[sp500_column],
            mode="lines",
            line={"color": "#f4f4f5", "width": 2.5},
            name="S&P 500",
        )
    )

    shading_frame = price_frame[["regime_name"]].copy()
    shading_frame["segment"] = shading_frame["regime_name"].ne(shading_frame["regime_name"].shift()).cumsum()
    for _, segment in shading_frame.groupby("segment"):
        regime_name = str(segment["regime_name"].iloc[0])
        fillcolor = hex_to_rgba(REGIME_COLORS.get(regime_name, "#64748b"), 0.16)
        figure.add_vrect(
            x0=segment.index.min(),
            x1=segment.index.max(),
            fillcolor=fillcolor,
            line_width=0,
            layer="below",
        )

    figure.update_layout(title="S&P 500 with Regime Timeline", xaxis_title="Date", yaxis_title="Price")
    return format_plotly(figure)


def build_probability_figure(probabilities: pd.DataFrame, labels: pd.DataFrame) -> go.Figure:
    """Build a stacked regime probability area chart."""
    figure = go.Figure()
    for column in probabilities.columns:
        regime_id = int(column.split("_")[-1])
        regime_name = (
            labels.loc[labels["regime_id"] == regime_id, "regime_name"].mode().iloc[0]
            if "regime_name" in labels.columns and (labels["regime_id"] == regime_id).any()
            else f"Regime {regime_id}"
        )
        color = REGIME_COLORS.get(str(regime_name), "#64748b")
        figure.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[column],
                mode="lines",
                stackgroup="one",
                name=str(regime_name),
                line={"width": 0.8, "color": color},
                fillcolor=hex_to_rgba(color, 0.38),
            )
        )
    figure.update_layout(title="Regime Probabilities", xaxis_title="Date", yaxis_title="Probability")
    return format_plotly(figure)


def build_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str) -> go.Figure:
    """Build a correlation heatmap for the provided matrix."""
    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title=title,
    )
    return format_plotly(figure)


def build_histogram(pnl_frame: pd.DataFrame, risk_metrics: pd.DataFrame) -> go.Figure:
    """Build a histogram of Monte Carlo portfolio P&L with VaR and CVaR markers."""
    figure = px.histogram(pnl_frame, x="portfolio_pnl", nbins=60, title="Monte Carlo P&L Distribution")
    for _, row in risk_metrics.iterrows():
        confidence = int(float(row["confidence_level"]) * 100)
        figure.add_vline(x=-float(row["var"]), line_dash="dash", line_color="#ff6b6b", annotation_text=f"VaR {confidence}")
        figure.add_vline(x=-float(row["cvar"]), line_dash="dot", line_color="#facc15", annotation_text=f"CVaR {confidence}")
    figure.update_layout(xaxis_title="Portfolio P&L", yaxis_title="Frequency")
    return format_plotly(figure)


def build_feature_heatmap(feature_matrix: pd.DataFrame) -> go.Figure:
    """Build a heatmap of the latest standardized feature values."""
    latest = feature_matrix.tail(30).T
    figure = px.imshow(latest, aspect="auto", color_continuous_scale="Tealrose", title="Latest Feature Regime Surface")
    return format_plotly(figure)


def build_drawdown_figure(cumulative_returns: pd.Series, title: str) -> go.Figure:
    """Build a drawdown chart from a cumulative return series."""
    running_peak = cumulative_returns.cummax()
    drawdown = cumulative_returns.div(running_peak).sub(1.0)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", fill="tozeroy", name="Drawdown"))
    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown")
    return format_plotly(figure)


def build_regime_distribution(labels: pd.DataFrame) -> go.Figure:
    """Build a donut chart showing regime share over the sample."""
    counts = labels["regime_name"].value_counts().rename_axis("regime_name").reset_index(name="count")
    figure = px.pie(
        counts,
        values="count",
        names="regime_name",
        hole=0.55,
        color="regime_name",
        color_discrete_map=REGIME_COLORS,
        title="Regime Mix",
    )
    return format_plotly(figure)


def build_regime_duration_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Build average and maximum spell duration statistics for each regime."""
    durations = labels.copy()
    durations["segment"] = durations["regime_name"].ne(durations["regime_name"].shift()).cumsum()
    spells = durations.groupby(["segment", "regime_name"]).size().reset_index(name="days")
    summary = spells.groupby("regime_name")["days"].agg(["count", "mean", "max"]).rename(
        columns={"count": "spells", "mean": "avg_days", "max": "max_days"}
    )
    return summary.sort_values("avg_days", ascending=False)
