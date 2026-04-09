"""Streamlit entrypoint for the quantrisk dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charting import (
    apply_zoom_range,
    build_correlation_heatmap,
    build_drawdown_figure,
    build_feature_heatmap,
    build_histogram,
    build_probability_figure,
    build_regime_distribution,
    build_regime_duration_table,
    build_regime_timeline_figure,
    filter_artifacts_by_date,
    format_plotly,
)
from dashboard.resources import get_data_dir, load_logo_data_uri
from dashboard.styling import inject_styles, is_light_theme
from quantrisk import ScenarioEngine, run_pipeline
from quantrisk.backtest import Backtester



def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    """Normalize portfolio weights so they sum to one when possible."""
    total_weight = sum(raw_weights.values())
    if total_weight == 0.0:
        count = len(raw_weights)
        return {asset: 1.0 / count for asset in raw_weights}
    return {asset: weight / total_weight for asset, weight in raw_weights.items()}


def render_chart_card(title: str, figure: go.Figure, key: str, zoomable_frame: pd.DataFrame | pd.Series | None = None) -> None:
    """Render a chart inside a structured dashboard card with optional zoom controls."""
    st.markdown(f"<div class='chart-shell'><h3>{title}</h3></div>", unsafe_allow_html=True)
    if zoomable_frame is not None:
        zoom_label = st.segmented_control(
            f"{title} zoom",
            options=["6M", "1Y", "3Y", "5Y", "Full"],
            default="3Y",
            key=f"{key}_zoom",
            label_visibility="collapsed",
        )
        figure = apply_zoom_range(figure, zoomable_frame, zoom_label)
        st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)
    st.plotly_chart(figure, use_container_width=True, key=f"{key}_plot")


def render_control_bar(page: str, lookback_label: str, n_regimes: int, portfolio_weights: dict[str, float]) -> None:
    """Render a compact top control summary so navigation does not depend on the sidebar."""
    largest_weight = max(portfolio_weights, key=portfolio_weights.get)
    items = [
        ("Page", page),
        ("Visual Lookback", lookback_label),
        ("Regimes", str(n_regimes)),
        ("Largest Weight", f"{largest_weight} {portfolio_weights[largest_weight]:.0%}"),
    ]
    cards = "".join(
        [
            f"<div class='control-pill'><div class='control-pill-label'>{label}</div><div class='control-pill-value'>{value}</div></div>"
            for label, value in items
        ]
    )
    st.markdown(f"<div class='control-bar'>{cards}</div>", unsafe_allow_html=True)


def render_section_caption(text: str) -> None:
    """Render a small section caption above a dashboard grid."""
    st.markdown(f"<div class='grid-section'><div class='grid-caption'>{text}</div></div>", unsafe_allow_html=True)


def render_control_panel() -> tuple[str, str, str, str, int, str, dict[str, float]]:
    """Render a dashboard-native control panel and return the selected values."""
    st.markdown("<div class='panel-title'>Controls</div>", unsafe_allow_html=True)
    if "dashboard_theme" not in st.session_state:
        st.session_state["dashboard_theme"] = "Dark"

    theme_value = st.segmented_control(
        "Dashboard theme",
        options=["Dark", "Light"],
        default=st.session_state["dashboard_theme"],
        key="dashboard_theme",
    )
    page = st.radio("Navigate", ["Regime Timeline", "Risk Metrics", "Scenario Stress Test", "Backtesting"], key="panel_page")
    start_date = st.date_input("Start date", value=pd.Timestamp("2010-01-01"), key="panel_start_date").strftime("%Y-%m-%d")
    end_date = st.date_input("End date", value=pd.Timestamp.today(), key="panel_end_date").strftime("%Y-%m-%d")
    n_regimes = st.slider("Number of regimes", min_value=2, max_value=6, value=4, key="panel_n_regimes")
    lookback_label = st.select_slider("Visual lookback", options=["1Y", "3Y", "5Y", "10Y", "Full"], value="Full", key="panel_lookback")

    st.caption("Portfolio weights for risk and scenario analytics")
    raw_weights = {
        "sp500": st.number_input("S&P 500", value=0.35, step=0.05, key="weight_sp500"),
        "msci_world": st.number_input("MSCI World", value=0.20, step=0.05, key="weight_msci_world"),
        "vix": st.number_input("VIX", value=0.05, step=0.05, key="weight_vix"),
        "gold": st.number_input("Gold", value=0.20, step=0.05, key="weight_gold"),
        "oil": st.number_input("Oil", value=0.10, step=0.05, key="weight_oil"),
        "eurusd": st.number_input("EUR/USD", value=0.15, step=0.05, key="weight_eurusd"),
    }
    return theme_value, page, start_date, end_date, n_regimes, lookback_label, normalize_weights(raw_weights)


def render_hero(current_regime: str, end_date: str, n_regimes: int, weights: dict[str, float]) -> None:
    """Render the top summary block."""
    largest_weight = max(weights, key=weights.get)
    logo_uri = load_logo_data_uri()
    chips = "".join(
        [
            f"<div class='section-chip'>Current regime: {current_regime}</div>",
            f"<div class='section-chip'>Model states: {n_regimes}</div>",
            f"<div class='section-chip'>Largest weight: {largest_weight} {weights[largest_weight]:.0%}</div>",
            f"<div class='section-chip'>Data through: {end_date}</div>",
        ]
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-row">
                <div class="hero-logo-wrap">
                    <img class="hero-logo" src="{logo_uri}" alt="Quantrisk logo" />
                </div>
                <div class="hero-body">
                    <div class="hero-eyebrow">Regime Analytics Workspace</div>
                    <div class="hero-title">Quantrisk Dashboard</div>
                    <div class="hero-copy">
                        Explore how market regimes reshape returns, cross-asset risk, stress losses,
                        and portfolio behavior across time.
                    </div>
                </div>
            </div>
            <div class="section-chip-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, copy: str) -> None:
    """Render a metric card using custom HTML."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(artifacts: dict[str, pd.DataFrame]) -> None:
    """Render top-level KPI cards shared by all pages."""
    main_data = artifacts["main_data"].dropna(subset=["regime_name"]).copy()
    current_regime = str(main_data["regime_name"].iloc[-1])
    current_probabilities = artifacts["regime_probabilities"].iloc[-1]
    dominant_probability = float(current_probabilities.max())
    risk_metrics = artifacts["risk_metrics"]
    current_risk = risk_metrics.loc[risk_metrics["regime_name"] == current_regime].iloc[0] if "regime_name" in risk_metrics.columns and current_regime in set(risk_metrics["regime_name"]) else None
    returns = artifacts["returns"]
    trailing_return = float((1.0 + returns["sp500"].tail(63)).prod() - 1.0) if "sp500" in returns.columns else np.nan

    columns = st.columns(4)
    payload = [
        ("Current Regime", current_regime, f"Model confidence {dominant_probability:.1%}"),
        ("S&P 500 3M Return", f"{trailing_return:.1%}" if pd.notna(trailing_return) else "n/a", "Recent market momentum"),
        ("Regime 95% VaR", f"{float(current_risk['historical_var_95']):.2%}" if current_risk is not None else "n/a", "Historical simulation"),
        ("Feature Count", str(artifacts["feature_matrix"].shape[1]), f"{artifacts['feature_matrix'].shape[0]} clean observations"),
    ]
    for column, item in zip(columns, payload, strict=False):
        with column:
            render_metric_card(*item)
@st.cache_data(show_spinner=False)
def load_pipeline_artifacts(
    start_date: str,
    end_date: str,
    portfolio_weight_items: tuple[tuple[str, float], ...],
    n_regimes: int,
) -> dict[str, pd.DataFrame]:
    """Run the cached pipeline and load supplementary parquet artifacts from disk."""
    portfolio_weights = dict(portfolio_weight_items)
    results = run_pipeline(
        start_date=start_date,
        end_date=end_date,
        portfolio_weights=portfolio_weights,
        n_regimes=n_regimes,
    )

    data_dir = get_data_dir()
    return {
        "main_data": pd.read_parquet(data_dir / "main_with_regimes.parquet"),
        "returns": pd.read_parquet(data_dir / "returns.parquet"),
        "risk_metrics": pd.read_parquet(data_dir / "risk_metrics.parquet"),
        "scenario_results": pd.read_parquet(data_dir / "scenario_results.parquet"),
        "feature_matrix": results["feature_matrix"],
        "regime_labels": results["regime_labels"],
        "regime_probabilities": results["regime_probabilities"],
    }


@st.cache_data(show_spinner=False)
def run_cached_backtest(regime_data: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    """Run and cache the proxy backtest for the current regime history."""
    backtester = Backtester(regime_data=regime_data)
    return backtester.run_backtest()


@st.cache_data(show_spinner=False)
def run_cached_monte_carlo(
    returns: pd.DataFrame,
    regime_labels: pd.DataFrame,
    portfolio_weight_items: tuple[tuple[str, float], ...],
    shock_items: tuple[tuple[str, float], ...],
    n_simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and cache Monte Carlo stress testing for interactive controls."""
    scenario_engine = ScenarioEngine(
        returns=returns,
        regime_data=regime_labels,
        portfolio_weights=dict(portfolio_weight_items),
    )
    return scenario_engine.run_monte_carlo(shocks=dict(shock_items), n_simulations=n_simulations)


def render_regime_timeline_page(artifacts: dict[str, pd.DataFrame]) -> None:
    """Render the regime timeline page."""
    price_frame = artifacts["main_data"].dropna(subset=["regime_name"]).copy()
    timeline_chart = build_regime_timeline_figure(price_frame)
    probability_chart = build_probability_figure(artifacts["regime_probabilities"], artifacts["regime_labels"])
    distribution_chart = build_regime_distribution(artifacts["regime_labels"])
    duration_table = build_regime_duration_table(artifacts["regime_labels"])
    feature_heatmap = build_feature_heatmap(artifacts["feature_matrix"])

    render_section_caption("Market State")
    top_left, top_right = st.columns(2)
    with top_left:
        render_chart_card("S&P 500 with Regime Timeline", timeline_chart, "timeline", price_frame)
    with top_right:
        render_chart_card("Regime Mix", distribution_chart, "distribution")
        st.dataframe(duration_table, use_container_width=True)

    render_section_caption("Model Surface")
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        render_chart_card("Regime Probabilities", probability_chart, "probabilities", artifacts["regime_probabilities"])
    with bottom_right:
        render_chart_card("Latest Feature Regime Surface", feature_heatmap, "feature_heatmap")
        st.download_button(
            "Download regime labels",
            data=artifacts["regime_labels"].to_csv().encode("utf-8"),
            file_name="regime_labels.csv",
            mime="text/csv",
        )


def render_risk_page(artifacts: dict[str, pd.DataFrame]) -> None:
    """Render the risk metrics page."""
    risk_metrics = artifacts["risk_metrics"].copy()
    regime_options = (
        sorted(risk_metrics["regime_name"].dropna().unique().tolist())
        if "regime_name" in risk_metrics.columns
        else sorted(risk_metrics["regime_id"].astype(str).unique().tolist())
    )
    selected_regime = st.selectbox("Current regime", regime_options)

    if "regime_name" in risk_metrics.columns:
        selected_table = risk_metrics.loc[risk_metrics["regime_name"] == selected_regime].copy()
    else:
        selected_table = risk_metrics.loc[risk_metrics["regime_id"].astype(str) == str(selected_regime)].copy()

    risk_columns = [
        column
        for column in selected_table.columns
        if column in {"regime_id", "regime_name"} or "var" in column or "shortfall" in column
    ]

    regime_labels = artifacts["regime_labels"]
    returns = artifacts["returns"]
    selected_mask = regime_labels["regime_name"] == selected_regime if "regime_name" in risk_metrics.columns else regime_labels["regime_id"].astype(str) == str(selected_regime)
    selected_returns = returns.loc[returns.index.intersection(regime_labels.index[selected_mask])]

    metric_snapshot = selected_table.iloc[0]
    columns = st.columns(3)
    payload = [
        ("95% Hist VaR", f"{float(metric_snapshot['historical_var_95']):.2%}", "Expected one-day loss threshold"),
        ("99% CVaR", f"{float(metric_snapshot['expected_shortfall_99']):.2%}", "Average tail loss in worst 1%"),
        ("Avg Correlation", f"{float(metric_snapshot['average_correlation']):.2f}", "Cross-asset co-movement inside regime"),
    ]
    for column, item in zip(columns, payload, strict=False):
        with column:
            render_metric_card(*item)

    render_section_caption("Risk Decomposition")
    left, right = st.columns(2)
    with left:
        st.dataframe(selected_table.loc[:, risk_columns], use_container_width=True)
        metric_plot = selected_table.loc[:, [column for column in risk_columns if column not in {"regime_id", "regime_name"}]].T
        st.bar_chart(metric_plot, use_container_width=True)

    with right:
        if selected_returns.empty:
            st.info("No return observations overlap the selected regime in the available sample.")
        else:
            correlation = selected_returns.corr()
            render_chart_card(f"Correlation Heatmap: {selected_regime}", build_correlation_heatmap(correlation, f"Correlation Heatmap: {selected_regime}"), "risk_heatmap")
            volatility_table = selected_returns.std().sort_values(ascending=False).rename("volatility").to_frame()
            st.dataframe(volatility_table, use_container_width=True)


def render_scenario_page(artifacts: dict[str, pd.DataFrame], portfolio_weights: dict[str, float]) -> None:
    """Render the scenario stress-test page."""
    returns = artifacts["returns"]
    regime_labels = artifacts["regime_labels"]
    scenario_engine = ScenarioEngine(returns=returns, regime_data=regime_labels, portfolio_weights=portfolio_weights)

    simulations = st.slider("Monte Carlo simulations", min_value=1000, max_value=20000, value=5000, step=1000)
    shocks: dict[str, float] = {}
    asset_columns = returns.columns.tolist()
    slider_columns = st.columns(min(3, max(1, len(asset_columns))))
    for index, asset in enumerate(asset_columns):
        with slider_columns[index % len(slider_columns)]:
            shocks[asset] = st.slider(
                f"{asset} shock",
                min_value=-0.50,
                max_value=0.50,
                value=0.0,
                step=0.01,
                format="%.2f",
            )

    if st.button("Run Monte Carlo", use_container_width=False):
        pnl_frame, risk_metrics = run_cached_monte_carlo(
            returns=returns,
            regime_labels=regime_labels,
            portfolio_weight_items=tuple(sorted(portfolio_weights.items())),
            shock_items=tuple(sorted(shocks.items())),
            n_simulations=simulations,
        )
        shocked_pnl = scenario_engine.apply_shocks(shocks)

        render_section_caption("Monte Carlo Stress")
        left, right = st.columns(2)
        with left:
            render_chart_card("Monte Carlo P&L Distribution", build_histogram(pnl_frame, risk_metrics), "scenario_hist")
        with right:
            render_metric_card("Instant Shock P&L", f"{shocked_pnl['shocked_portfolio_return']:.2%}", "Weighted one-shot shock impact")
            render_metric_card("Simulated Mean P&L", f"{float(pnl_frame['portfolio_pnl'].mean()):.2%}", "Average one-day Monte Carlo outcome")
            render_metric_card("Tail Skew", f"{float(pnl_frame['portfolio_pnl'].skew()):.2f}", "Distribution asymmetry")
            st.dataframe(risk_metrics, use_container_width=True)
            st.download_button(
                "Download Monte Carlo draws",
                data=pnl_frame.to_csv(index=False).encode("utf-8"),
                file_name="monte_carlo_pnl.csv",
                mime="text/csv",
            )

    st.subheader("Historical Replay")
    replay_windows = {
        "2008 GFC": ("2008-09-01", "2009-03-31"),
        "2020 COVID": ("2020-02-01", "2020-04-30"),
        "2022 Rate Hike Cycle": ("2022-01-01", "2022-12-31"),
    }
    replay_label = st.selectbox("Historical scenario", list(replay_windows))
    replay_start, replay_end = replay_windows[replay_label]

    try:
        replay = scenario_engine.historical_replay(replay_start, replay_end)
        replay_chart = px.line(replay, x=replay.index, y="cumulative_return", title=f"Historical Replay: {replay_label}")
        replay_chart.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
        replay_chart = format_plotly(replay_chart)
        render_section_caption("Historical Replay")
        left, right = st.columns(2)
        with left:
            render_chart_card(f"Historical Replay: {replay_label}", replay_chart, "replay_chart", replay)
        with right:
            st.dataframe(replay.describe().T, use_container_width=True)
            render_chart_card(f"{replay_label} Drawdown", build_drawdown_figure((1.0 + replay["portfolio_return"]).cumprod(), f"{replay_label} Drawdown"), "replay_drawdown", replay)
    except ValueError as error:
        st.info(str(error))


def render_backtest_page(artifacts: dict[str, pd.DataFrame]) -> None:
    """Render the backtesting page."""
    comparison, figure = run_cached_backtest(artifacts["regime_labels"])
    strategy_lead = float(comparison.loc["Regime-Aware Strategy", "annualized_return"] - comparison.loc["Static 60/40 Benchmark", "annualized_return"])
    drawdown_delta = float(comparison.loc["Regime-Aware Strategy", "maximum_drawdown"] - comparison.loc["Static 60/40 Benchmark", "maximum_drawdown"])

    columns = st.columns(3)
    payload = [
        ("Return Edge", f"{strategy_lead:.2%}", "Annualized return difference versus benchmark"),
        ("Drawdown Delta", f"{drawdown_delta:.2%}", "Regime-aware minus benchmark drawdown"),
        ("Best Sharpe", f"{comparison['sharpe_ratio'].max():.2f}", "Highest risk-adjusted score on this run"),
    ]
    for column, item in zip(columns, payload, strict=False):
        with column:
            render_metric_card(*item)

    render_section_caption("Performance")
    render_chart_card("Cumulative Returns", figure, "backtest_returns")

    render_section_caption("Diagnostics")
    left, right = st.columns(2)
    with left:
        st.dataframe(comparison, use_container_width=True)
        st.bar_chart(comparison[["annualized_return", "sharpe_ratio"]], use_container_width=True)

    with right:
        backtester = Backtester(regime_data=artifacts["regime_labels"])
        strategy_path = backtester.compute_portfolio_path(benchmark=False)
        benchmark_path = backtester.compute_portfolio_path(benchmark=True)
        drawdown_chart = go.Figure()
        for name, path, color in (
            ("Regime-Aware Strategy", strategy_path, "#7c83ff"),
            ("Static 60/40 Benchmark", benchmark_path, "#ff7f50"),
        ):
            running_peak = path["cumulative_return"].cummax()
            drawdown = path["cumulative_return"].div(running_peak).sub(1.0)
            drawdown_chart.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name=name, line={"color": color}))
        drawdown_chart.update_layout(title="Strategy Drawdowns", xaxis_title="Date", yaxis_title="Drawdown")
        render_chart_card("Strategy Drawdowns", format_plotly(drawdown_chart), "backtest_drawdown", strategy_path["cumulative_return"])


def main() -> None:
    """Render the multi-page quantrisk dashboard."""
    st.set_page_config(page_title="Quantrisk", layout="wide", initial_sidebar_state="expanded")
    if "dashboard_theme" not in st.session_state:
        st.session_state["dashboard_theme"] = "Dark"

    light_theme = is_light_theme()
    inject_styles(light_theme)

    toolbar_left, toolbar_right = st.columns([1, 6], gap="small")
    with toolbar_left:
        with st.popover("Controls", use_container_width=True):
            theme_value, page, start_date, end_date, n_regimes, lookback_label, portfolio_weights = render_control_panel()
    with toolbar_right:
        st.markdown("<div class='toolbar-spacer'></div>", unsafe_allow_html=True)

    theme_value = st.session_state.get("dashboard_theme", "Dark")
    light_theme = theme_value == "Light"
    inject_styles(light_theme)
    page = st.session_state.get("panel_page", "Regime Timeline")
    start_date = pd.Timestamp(st.session_state.get("panel_start_date", pd.Timestamp("2010-01-01"))).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(st.session_state.get("panel_end_date", pd.Timestamp.today())).strftime("%Y-%m-%d")
    n_regimes = int(st.session_state.get("panel_n_regimes", 4))
    lookback_label = st.session_state.get("panel_lookback", "Full")
    portfolio_weights = normalize_weights(
        {
            "sp500": float(st.session_state.get("weight_sp500", 0.35)),
            "msci_world": float(st.session_state.get("weight_msci_world", 0.20)),
            "vix": float(st.session_state.get("weight_vix", 0.05)),
            "gold": float(st.session_state.get("weight_gold", 0.20)),
            "oil": float(st.session_state.get("weight_oil", 0.10)),
            "eurusd": float(st.session_state.get("weight_eurusd", 0.15)),
        }
    )

    try:
        artifacts = load_pipeline_artifacts(
            start_date=start_date,
            end_date=end_date,
            portfolio_weight_items=tuple(sorted(portfolio_weights.items())),
            n_regimes=n_regimes,
        )
    except Exception as error:
        st.error(f"Pipeline failed: {error}")
        st.stop()

    lookback_days_map = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "10Y": 365 * 10}
    if lookback_label != "Full":
        artifacts = filter_artifacts_by_date(artifacts, lookback_days_map[lookback_label])

    current_regime = str(artifacts["main_data"].dropna(subset=["regime_name"])["regime_name"].iloc[-1])
    render_hero(current_regime=current_regime, end_date=end_date, n_regimes=n_regimes, weights=portfolio_weights)
    render_control_bar(page=page, lookback_label=lookback_label, n_regimes=n_regimes, portfolio_weights=portfolio_weights)
    render_kpi_cards(artifacts)

    if page == "Regime Timeline":
        render_regime_timeline_page(artifacts)
    elif page == "Risk Metrics":
        render_risk_page(artifacts)
    elif page == "Scenario Stress Test":
        render_scenario_page(artifacts, portfolio_weights)
    else:
        render_backtest_page(artifacts)


if __name__ == "__main__":
    main()
