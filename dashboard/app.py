"""Streamlit entrypoint for the quantrisk dashboard."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantrisk import ScenarioEngine, run_pipeline
from quantrisk.backtest import Backtester

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


def get_data_dir() -> Path:
    """Return the local data directory used by the pipeline."""
    return Path(__file__).resolve().parents[1] / "data"


def get_assets_dir() -> Path:
    """Return the dashboard assets directory."""
    return Path(__file__).resolve().parent / "assets"


def load_logo_data_uri() -> str:
    """Load the project logo SVG and return it as a data URI."""
    logo_path = get_assets_dir() / "quantrisk_logo.svg"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def is_light_theme() -> bool:
    """Return whether the dashboard should render in light theme."""
    theme_choice = st.session_state.get("dashboard_theme", "Dark")
    if theme_choice == "System":
        return st.get_option("theme.base") == "light"
    return theme_choice == "Light"


def inject_styles(light_theme: bool) -> None:
    """Apply a richer visual design layer to the dashboard."""
    page_background = "#f5f7fb" if light_theme else "#050a14"
    text_color = "#0f172a" if light_theme else "#f8fafc"
    muted_text = "rgba(15,23,42,0.72)" if light_theme else "rgba(255,255,255,0.72)"
    muted_copy = "rgba(15,23,42,0.78)" if light_theme else "rgba(255,255,255,0.75)"
    surface = "#ffffff" if light_theme else "rgba(255,255,255,0.04)"
    surface_strong = "#ffffff" if light_theme else "rgba(255,255,255,0.06)"
    border = "rgba(15,23,42,0.08)" if light_theme else "rgba(255,255,255,0.08)"
    sidebar_background = "linear-gradient(180deg, #11284b 0%, #0d1f38 100%)" if light_theme else "linear-gradient(180deg, #101826 0%, #161927 100%)"
    app_background = (
        f"linear-gradient(180deg, #ffffff 0%, {page_background} 100%)"
        if light_theme
        else "radial-gradient(circle at top left, rgba(38, 83, 117, 0.18), transparent 25%),"
        "radial-gradient(circle at top right, rgba(196, 61, 61, 0.14), transparent 22%),"
        "linear-gradient(180deg, #07111f 0%, #050a14 100%)"
    )
    css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: __APP_BACKGROUND__;
        }
        [data-testid="stHeader"] {
            background: __HEADER_BG__;
            border-bottom: 1px solid __HEADER_BORDER__;
        }
        [data-testid="stHeader"] * {
            color: __TEXT_COLOR__;
        }
        .stApp {
            background: __APP_BACKGROUND__;
            overflow-x: hidden;
            color: __TEXT_COLOR__;
        }
        .stApp::before, .stApp::after {
            content: "";
            position: fixed;
            width: 32rem;
            height: 32rem;
            border-radius: 999px;
            filter: blur(60px);
            opacity: __ORB_OPACITY__;
            pointer-events: none;
            z-index: 0;
            animation: floatOrb 14s ease-in-out infinite;
        }
        .stApp::before {
            background: #7c83ff;
            top: -8rem;
            right: -6rem;
        }
        .stApp::after {
            background: #2dd4bf;
            bottom: -10rem;
            left: -8rem;
            animation-delay: -7s;
        }
        [data-testid="stSidebar"] {
            background: __SIDEBAR_BACKGROUND__;
            border-right: 1px solid __BORDER__;
            min-width: 21rem !important;
            max-width: 21rem !important;
            width: 21rem !important;
            transform: translateX(0) !important;
            margin-left: 0 !important;
            visibility: visible !important;
            display: block !important;
            position: sticky;
            left: 0;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            display: none !important;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        button[kind="header"], [data-testid="collapsedControl"] {
            display: none !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"] * {
            color: __SIDEBAR_TEXT__;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label {
            color: __SIDEBAR_TEXT__;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="input"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background: __SIDEBAR_INPUT_BG__ !important;
            color: __SIDEBAR_TEXT__ !important;
            border-color: __SIDEBAR_INPUT_BORDER__ !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            box-shadow: 0 0 0 2px rgba(255,255,255,0.16);
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label {
            padding: 0.38rem 0.55rem;
            border-radius: 12px;
            transition: background 150ms ease;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
            background: rgba(255,255,255,0.08);
        }
        .hero-card {
            padding: 1.45rem 1.5rem;
            border-radius: 18px;
            background: __HERO_BG__;
            border: 1px solid __BORDER__;
            box-shadow: __CARD_SHADOW__;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            animation: riseIn 0.85s cubic-bezier(.22,1,.36,1);
        }
        .hero-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.08) 45%, transparent 70%);
            transform: translateX(-120%);
            animation: shimmer 5.5s ease-in-out infinite;
        }
        .hero-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #8fb3d9;
            margin-bottom: 0.45rem;
        }
        .hero-row {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .hero-logo-wrap {
            width: 5rem;
            height: 5rem;
            flex: 0 0 5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 22px;
            background: __LOGO_WRAP_BG__;
            border: 1px solid __BORDER__;
            backdrop-filter: blur(10px);
            animation: pulseGlow 4s ease-in-out infinite;
        }
        .hero-logo {
            width: 4rem;
            height: 4rem;
        }
        .hero-body {
            flex: 1;
            min-width: 0;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.45rem;
            color: __TEXT_COLOR__;
        }
        .hero-copy {
            color: __MUTED_COPY__;
            font-size: 0.98rem;
        }
        .section-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.95rem 0 0.2rem 0;
        }
        .section-chip {
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            background: __CHIP_BG__;
            border: 1px solid __BORDER__;
            font-size: 0.82rem;
            color: __TEXT_COLOR__;
        }
        .insight-card {
            padding: 1.1rem 1.1rem;
            border-radius: 16px;
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            min-height: 138px;
            animation: riseIn 0.9s cubic-bezier(.22,1,.36,1);
            transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
            box-shadow: __CARD_SHADOW_SOFT__;
        }
        .insight-card:hover {
            transform: translateY(-4px);
            border-color: rgba(124,131,255,0.35);
            background: __SURFACE_STRONG__;
        }
        .insight-label {
            color: #8fb3d9;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .insight-value {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0.25rem 0;
            color: __TEXT_COLOR__;
        }
        .insight-copy {
            color: __MUTED_TEXT__;
            font-size: 0.9rem;
        }
        .control-bar {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 0 0 1.9rem 0;
        }
        .dashboard-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .panel-shell {
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            border-radius: 18px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: __CARD_SHADOW_SOFT__;
        }
        .panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: __TEXT_COLOR__;
            margin-bottom: 0.8rem;
        }
        .toolbar-spacer {
            height: 0.25rem;
        }
        .control-pill {
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            border-radius: 16px;
            padding: 1.05rem 1.1rem;
            box-shadow: __CARD_SHADOW_SOFT__;
        }
        .control-pill-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8fb3d9;
            margin-bottom: 0.35rem;
        }
        .control-pill-value {
            font-size: 1.02rem;
            font-weight: 700;
            color: __TEXT_COLOR__;
        }
        .grid-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: __TEXT_COLOR__;
            margin: 1rem 0 0.65rem 0;
        }
        .chart-shell {
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: __CARD_SHADOW_SOFT__;
            margin-bottom: 1.2rem;
        }
        .grid-section {
            margin-top: 1.4rem;
            margin-bottom: 1.2rem;
        }
        .grid-caption {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8fb3d9;
            margin: 0 0 0.8rem 0.1rem;
        }
        .chart-shell h3 {
            margin: 0 0 0.95rem 0;
            font-size: 1rem;
            color: __TEXT_COLOR__;
        }
        [data-testid="stMetric"] {
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            animation: riseIn 0.9s cubic-bezier(.22,1,.36,1);
            box-shadow: __CARD_SHADOW_SOFT__;
        }
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
            animation: riseIn 1s cubic-bezier(.22,1,.36,1);
        }
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            background: __SURFACE__;
            border: 1px solid __BORDER__;
            border-radius: 18px;
            padding: 0.3rem;
            box-shadow: __CARD_SHADOW_SOFT__;
        }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stDataFrame"] > div {
            border-radius: 14px;
            overflow: hidden;
        }
        .block-container {
            padding-top: 2.3rem;
            padding-bottom: 4rem;
            max-width: 1550px;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
            margin-top: 0.75rem;
            margin-bottom: 1.2rem;
        }
        div[data-testid="column"] {
            padding-top: 0.35rem;
            padding-bottom: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 1.4rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }
        [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stSelectSlider,
        [data-testid="stSidebar"] .stRadio {
            margin-bottom: 1rem;
        }
        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes floatOrb {
            0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
            50% { transform: translate3d(0, 18px, 0) scale(1.08); }
        }
        @keyframes shimmer {
            0%, 18% { transform: translateX(-120%); }
            45% { transform: translateX(120%); }
            100% { transform: translateX(120%); }
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 rgba(124,131,255,0.0); transform: scale(1); }
            50% { box-shadow: 0 0 32px rgba(124,131,255,0.18); transform: scale(1.02); }
        }
        </style>
        """
    css = (
        css.replace("__APP_BACKGROUND__", app_background)
        .replace("__HEADER_BG__", "rgba(255,255,255,0.92)" if light_theme else "rgba(5,10,20,0.92)")
        .replace("__HEADER_BORDER__", "rgba(15,23,42,0.08)" if light_theme else "rgba(255,255,255,0.06)")
        .replace("__TEXT_COLOR__", text_color)
        .replace("__ORB_OPACITY__", str(0.10 if light_theme else 0.18))
        .replace("__SIDEBAR_BACKGROUND__", sidebar_background)
        .replace("__SIDEBAR_TEXT__", "#f8fafc" if light_theme else "#dbe7f5")
        .replace("__SIDEBAR_INPUT_BG__", "rgba(255,255,255,0.08)" if light_theme else "rgba(255,255,255,0.06)")
        .replace("__SIDEBAR_INPUT_BORDER__", "rgba(255,255,255,0.16)" if light_theme else "rgba(255,255,255,0.12)")
        .replace("__BORDER__", border)
        .replace("__SURFACE__", surface)
        .replace("__SURFACE_STRONG__", surface_strong)
        .replace("__MUTED_TEXT__", muted_text)
        .replace("__MUTED_COPY__", muted_copy)
        .replace("__HERO_BG__", "linear-gradient(180deg, #ffffff 0%, #fbfcff 100%)" if light_theme else f"linear-gradient(135deg, {surface_strong}, {surface})")
        .replace("__LOGO_WRAP_BG__", "#f8fbff" if light_theme else surface)
        .replace("__CHIP_BG__", "#f7f9fd" if light_theme else surface)
        .replace("__CARD_SHADOW__", "0 10px 28px rgba(15,23,42,0.08)" if light_theme else "0 18px 50px rgba(0,0,0,0.22)")
        .replace("__CARD_SHADOW_SOFT__", "0 8px 22px rgba(15,23,42,0.05)" if light_theme else "none")
    )
    st.markdown(css, unsafe_allow_html=True)


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    """Normalize portfolio weights so they sum to one when possible."""
    total_weight = sum(raw_weights.values())
    if total_weight == 0.0:
        count = len(raw_weights)
        return {asset: 1.0 / count for asset in raw_weights}
    return {asset: weight / total_weight for asset, weight in raw_weights.items()}


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

    if isinstance(frame, pd.Series):
        index = frame.index
    else:
        index = frame.index

    if len(index) == 0:
        return figure

    end_date = pd.to_datetime(index.max())
    zoom_map = {"6M": 183, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
    start_date = end_date - pd.Timedelta(days=zoom_map[zoom_label])
    figure.update_xaxes(range=[start_date, end_date])
    return figure


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


def render_control_panel() -> tuple[str, str, str, int, dict[str, float]]:
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


def build_regime_timeline_figure(price_frame: pd.DataFrame) -> go.Figure:
    """Create an S&P 500 price chart with background shading by regime."""
    price_column = get_sp500_price_column(price_frame)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=price_frame.index,
            y=price_frame[price_column],
            mode="lines",
            line={"color": "#f8fafc", "width": 2.3},
            name="S&P 500",
        )
    )

    regime_changes = price_frame["regime_name"].ne(price_frame["regime_name"].shift()).cumsum()
    for _, segment in price_frame.groupby(regime_changes):
        regime_name = str(segment["regime_name"].iloc[0])
        figure.add_vrect(
            x0=segment.index.min(),
            x1=segment.index.max(),
            fillcolor=hex_to_rgba(REGIME_COLORS.get(regime_name, "#777777"), 0.2),
            line_width=0,
            layer="below",
        )

    figure.update_layout(title="S&P 500 with Regime Timeline", xaxis_title="Date", yaxis_title="Price")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return format_plotly(figure)


def build_probability_figure(probabilities: pd.DataFrame, labels: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart of regime probabilities over time."""
    figure = go.Figure()
    regime_name_map = (
        labels.dropna(subset=["regime_id", "regime_name"])
        .drop_duplicates(subset=["regime_id"])
        .set_index("regime_id")["regime_name"]
        .to_dict()
    )

    for column in probabilities.columns:
        regime_id = int(column.split("_")[-1])
        regime_name = regime_name_map.get(regime_id, f"Regime {regime_id}")
        figure.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[column],
                mode="lines",
                stackgroup="one",
                name=regime_name,
                line={"width": 0.9, "color": REGIME_COLORS.get(regime_name, "#7f8c8d")},
                fillcolor=hex_to_rgba(REGIME_COLORS.get(regime_name, "#777777"), 0.4),
                hovertemplate="%{x|%Y-%m-%d}<br>Probability=%{y:.2%}<extra></extra>",
            )
        )

    figure.update_layout(title="Regime Probabilities", xaxis_title="Date", yaxis_title="Probability")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return format_plotly(figure)


def build_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str) -> go.Figure:
    """Create a Plotly heatmap for the supplied correlation matrix."""
    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title=title,
        aspect="auto",
    )
    return format_plotly(figure)


def build_histogram(pnl_frame: pd.DataFrame, risk_metrics: pd.DataFrame) -> go.Figure:
    """Create a histogram of simulated P&L with VaR and CVaR markers."""
    figure = px.histogram(pnl_frame, x="portfolio_pnl", nbins=60, title="Monte Carlo P&L Distribution")
    colors = {0.95: "#d62728", 0.99: "#9467bd"}

    for _, row in risk_metrics.iterrows():
        confidence_level = float(row["confidence_level"])
        var_level = -float(row["var"])
        cvar_level = -float(row["cvar"])
        line_color = colors.get(confidence_level, "#f8fafc")
        figure.add_vline(x=var_level, line_dash="dash", line_color=line_color)
        figure.add_vline(x=cvar_level, line_dash="dot", line_color=line_color)
        figure.add_annotation(x=var_level, y=0.97, yref="paper", text=f"VaR {int(confidence_level * 100)}", showarrow=False)
        figure.add_annotation(x=cvar_level, y=0.90, yref="paper", text=f"CVaR {int(confidence_level * 100)}", showarrow=False)

    figure.update_layout(xaxis_title="Portfolio P&L", yaxis_title="Count")
    return format_plotly(figure)


def build_feature_heatmap(feature_matrix: pd.DataFrame) -> go.Figure:
    """Create a heatmap for the latest normalized feature values."""
    latest = feature_matrix.tail(40).T
    figure = px.imshow(latest, color_continuous_scale="Tealrose", aspect="auto", title="Latest Feature Regime Surface")
    return format_plotly(figure)


def build_drawdown_figure(cumulative_returns: pd.Series, title: str) -> go.Figure:
    """Create a drawdown chart from cumulative returns."""
    running_peak = cumulative_returns.cummax()
    drawdown = cumulative_returns.div(running_peak).sub(1.0)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line={"color": "#ff7f50", "width": 1.6}, name="Drawdown")
    )
    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown")
    return format_plotly(figure)


def build_regime_distribution(labels: pd.DataFrame) -> go.Figure:
    """Create a regime composition donut chart."""
    counts = labels["regime_name"].value_counts().rename_axis("regime_name").reset_index(name="days")
    figure = px.pie(
        counts,
        values="days",
        names="regime_name",
        hole=0.62,
        color="regime_name",
        color_discrete_map=REGIME_COLORS,
        title="Regime Mix",
    )
    return format_plotly(figure)


def build_regime_duration_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Compute duration statistics for each regime."""
    durations = labels.copy()
    durations["segment"] = durations["regime_name"].ne(durations["regime_name"].shift()).cumsum()
    spells = durations.groupby(["segment", "regime_name"]).size().reset_index(name="days")
    summary = spells.groupby("regime_name")["days"].agg(["count", "mean", "max"]).rename(
        columns={"count": "spells", "mean": "avg_days", "max": "max_days"}
    )
    return summary.sort_values("avg_days", ascending=False)


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


def filter_artifacts_by_date(artifacts: dict[str, pd.DataFrame], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Filter date-indexed artifacts to a recent window for faster visuals."""
    filtered = artifacts.copy()
    end_date = pd.to_datetime(artifacts["main_data"].index.max())
    start_cutoff = end_date - pd.Timedelta(days=lookback_days)
    for key in ("main_data", "returns", "feature_matrix", "regime_labels", "regime_probabilities"):
        filtered[key] = filtered[key].loc[filtered[key].index >= start_cutoff].copy()
    return filtered


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
