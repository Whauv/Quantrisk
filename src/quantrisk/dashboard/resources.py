"""Filesystem and asset helpers for the dashboard."""

from __future__ import annotations

import base64
from pathlib import Path


def get_data_dir() -> Path:
    """Return the local data directory used by the pipeline."""
    return Path(__file__).resolve().parents[3] / "data"


def get_assets_dir() -> Path:
    """Return the dashboard assets directory."""
    return Path(__file__).resolve().parent / "assets"


def load_logo_data_uri() -> str:
    """Load the project logo SVG and return it as a data URI."""
    logo_path = get_assets_dir() / "quantrisk_logo.svg"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
