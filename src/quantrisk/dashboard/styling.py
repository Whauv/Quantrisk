"""Theme and style helpers for the dashboard."""

from __future__ import annotations

import streamlit as st


def is_light_theme() -> bool:
    """Return whether the dashboard should render in light theme."""
    theme_choice = st.session_state.get("dashboard_theme", "Dark")
    if theme_choice == "System":
        return st.get_option("theme.base") == "light"
    return theme_choice == "Light"


def inject_styles(light_theme: bool) -> None:
    """Apply the custom visual design layer to the dashboard."""
    page_background = "#f5f7fb" if light_theme else "#050a14"
    text_color = "#0f172a" if light_theme else "#f8fafc"
    muted_text = "rgba(15,23,42,0.72)" if light_theme else "rgba(255,255,255,0.72)"
    muted_copy = "rgba(15,23,42,0.78)" if light_theme else "rgba(255,255,255,0.75)"
    surface = "#ffffff" if light_theme else "rgba(255,255,255,0.04)"
    surface_strong = "#ffffff" if light_theme else "rgba(255,255,255,0.06)"
    border = "rgba(15,23,42,0.08)" if light_theme else "rgba(255,255,255,0.08)"
    sidebar_background = (
        "linear-gradient(180deg, #11284b 0%, #0d1f38 100%)"
        if light_theme
        else "linear-gradient(180deg, #101826 0%, #161927 100%)"
    )
    app_background = (
        f"linear-gradient(180deg, #ffffff 0%, {page_background} 100%)"
        if light_theme
        else ",".join(
            [
                "radial-gradient(circle at top left, rgba(38, 83, 117, 0.18), transparent 25%)",
                "radial-gradient(circle at top right, rgba(196, 61, 61, 0.14), transparent 22%)",
                "linear-gradient(180deg, #07111f 0%, #050a14 100%)",
            ]
        )
    )
    css = """
        <style>
        [data-testid="stAppViewContainer"] { background: __APP_BACKGROUND__; }
        [data-testid="stHeader"] { background: __HEADER_BG__; border-bottom: 1px solid __HEADER_BORDER__; }
        [data-testid="stHeader"] * { color: __TEXT_COLOR__; }
        .stApp { background: __APP_BACKGROUND__; overflow-x: hidden; color: __TEXT_COLOR__; }
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
        .stApp::before { background: #7c83ff; top: -8rem; right: -6rem; }
        .stApp::after { background: #2dd4bf; bottom: -10rem; left: -8rem; animation-delay: -7s; }
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
        [data-testid="stToolbar"] { display: none; }
        button[kind="header"], [data-testid="collapsedControl"] {
            display: none !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"] * { color: __SIDEBAR_TEXT__; }
        [data-testid="stSidebar"] [data-baseweb="radio"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label { color: __SIDEBAR_TEXT__; }
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="input"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background: __SIDEBAR_INPUT_BG__ !important;
            color: __SIDEBAR_TEXT__ !important;
            border-color: __SIDEBAR_INPUT_BORDER__ !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] { box-shadow: 0 0 0 2px rgba(255,255,255,0.16); }
        [data-testid="stSidebar"] [data-baseweb="radio"] label {
            padding: 0.38rem 0.55rem;
            border-radius: 12px;
            transition: background 150ms ease;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label:hover { background: rgba(255,255,255,0.08); }
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
        .hero-eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; color: #8fb3d9; margin-bottom: 0.45rem; }
        .hero-row { display: flex; gap: 1rem; align-items: center; }
        .hero-logo-wrap {
            width: 5rem; height: 5rem; flex: 0 0 5rem; display: flex; align-items: center; justify-content: center;
            border-radius: 22px; background: __LOGO_WRAP_BG__; border: 1px solid __BORDER__; backdrop-filter: blur(10px);
            animation: pulseGlow 4s ease-in-out infinite;
        }
        .hero-logo { width: 4rem; height: 4rem; }
        .hero-body { flex: 1; min-width: 0; }
        .hero-title { font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.45rem; color: __TEXT_COLOR__; }
        .hero-copy { color: __MUTED_COPY__; font-size: 0.98rem; }
        .section-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.95rem 0 0.2rem 0; }
        .section-chip {
            padding: 0.38rem 0.75rem; border-radius: 999px; background: __CHIP_BG__; border: 1px solid __BORDER__;
            font-size: 0.82rem; color: __TEXT_COLOR__;
        }
        .insight-card {
            padding: 1.1rem 1.1rem; border-radius: 16px; background: __SURFACE__; border: 1px solid __BORDER__;
            min-height: 138px; animation: riseIn 0.9s cubic-bezier(.22,1,.36,1);
            transition: transform 180ms ease, border-color 180ms ease, background 180ms ease; box-shadow: __CARD_SHADOW_SOFT__;
        }
        .insight-card:hover { transform: translateY(-4px); border-color: rgba(124,131,255,0.35); background: __SURFACE_STRONG__; }
        .insight-label { color: #8fb3d9; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .insight-value { font-size: 1.7rem; font-weight: 800; margin: 0.25rem 0; color: __TEXT_COLOR__; }
        .insight-copy { color: __MUTED_TEXT__; font-size: 0.9rem; }
        .control-bar { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 0 0 1.9rem 0; }
        .dashboard-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: 1.25rem; }
        .panel-shell { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 18px; padding: 1rem 1rem 0.4rem 1rem; box-shadow: __CARD_SHADOW_SOFT__; }
        .panel-title { font-size: 1.05rem; font-weight: 800; color: __TEXT_COLOR__; margin-bottom: 0.8rem; }
        .toolbar-spacer { height: 0.25rem; }
        .control-pill { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 16px; padding: 1.05rem 1.1rem; box-shadow: __CARD_SHADOW_SOFT__; }
        .control-pill-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8fb3d9; margin-bottom: 0.35rem; }
        .control-pill-value { font-size: 1.02rem; font-weight: 700; color: __TEXT_COLOR__; }
        .grid-title { font-size: 1.1rem; font-weight: 700; color: __TEXT_COLOR__; margin: 1rem 0 0.65rem 0; }
        .chart-shell { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 18px; padding: 1rem 1rem 0.85rem 1rem; box-shadow: __CARD_SHADOW_SOFT__; margin-bottom: 1.2rem; }
        .grid-section { margin-top: 1.4rem; margin-bottom: 1.2rem; }
        .grid-caption { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8fb3d9; margin: 0 0 0.8rem 0.1rem; }
        .chart-shell h3 { margin: 0 0 0.95rem 0; font-size: 1rem; color: __TEXT_COLOR__; }
        [data-testid="stMetric"] { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 16px; padding: 1rem 1.1rem; animation: riseIn 0.9s cubic-bezier(.22,1,.36,1); box-shadow: __CARD_SHADOW_SOFT__; }
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { animation: riseIn 1s cubic-bezier(.22,1,.36,1); }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
            background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 18px; padding: 0.3rem; box-shadow: __CARD_SHADOW_SOFT__;
        }
        [data-testid="stPlotlyChart"] > div, [data-testid="stDataFrame"] > div { border-radius: 14px; overflow: hidden; }
        .block-container { padding-top: 2.3rem; padding-bottom: 4rem; max-width: 1550px; }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) { margin-top: 0.75rem; margin-bottom: 1.2rem; }
        div[data-testid="column"] { padding-top: 0.35rem; padding-bottom: 0.75rem; }
        div[data-testid="stHorizontalBlock"] { gap: 1.4rem; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
        [data-testid="stSidebar"] .stNumberInput, [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stSelectSlider,
        [data-testid="stSidebar"] .stRadio { margin-bottom: 1rem; }
        @keyframes riseIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes floatOrb { 0%, 100% { transform: translate3d(0, 0, 0) scale(1); } 50% { transform: translate3d(0, 18px, 0) scale(1.08); } }
        @keyframes shimmer { 0%, 18% { transform: translateX(-120%); } 45% { transform: translateX(120%); } 100% { transform: translateX(120%); } }
        @keyframes pulseGlow { 0%, 100% { box-shadow: 0 0 0 rgba(124,131,255,0.0); transform: scale(1); } 50% { box-shadow: 0 0 32px rgba(124,131,255,0.18); transform: scale(1.02); } }
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
