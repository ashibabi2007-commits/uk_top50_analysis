# utils/helpers.py

import pandas as pd
import numpy as np
import re

# ─────────────────────────────────────────────
# COLOR PALETTE & BRAND CONSTANTS
# ─────────────────────────────────────────────
ATLANTIC_COLORS = {
    "primary":     "#0A3161",   # Deep Atlantic Blue
    "secondary":   "#C8102E",   # Atlantic Red
    "accent":      "#FFD700",   # Gold
    "light":       "#F0F4FA",
    "dark":        "#0D0D0D",
    "success":     "#2ECC71",
    "warning":     "#F39C12",
    "explicit":    "#E74C3C",
    "clean":       "#27AE60",
    "solo":        "#3498DB",
    "collab":      "#9B59B6",
    "single":      "#1ABC9C",
    "album":       "#E67E22",
}

CHART_TEMPLATE = "plotly_dark"

RANK_GROUPS = {
    "Top 10":    (1,  10),
    "Top 11-25": (11, 25),
    "Top 26-50": (26, 50),
}

# ─────────────────────────────────────────────
# ARTIST PARSING UTILITIES
# ─────────────────────────────────────────────
def split_artists(artist_str: str) -> list[str]:
    """
    Split a multi-artist string into individual artists.
    Handles delimiters: &, feat., ft., featuring, ,
    Returns a cleaned list of unique artist names.
    """
    if pd.isna(artist_str) or not isinstance(artist_str, str):
        return []

    # Normalize delimiters
    cleaned = re.sub(
        r'\s*(feat\.|ft\.|featuring|,|&)\s*',
        '|',
        artist_str,
        flags=re.IGNORECASE
    )
    artists = [a.strip() for a in cleaned.split('|') if a.strip()]
    return list(dict.fromkeys(artists))   # preserve order, remove dups


def normalize_artist_name(name: str) -> str:
    """Lowercase, strip whitespace, remove special chars for matching."""
    if pd.isna(name):
        return ""
    return re.sub(r'\s+', ' ', str(name).strip().lower())


def get_rank_group(position: int) -> str:
    """Map a chart position to a named rank group."""
    for label, (lo, hi) in RANK_GROUPS.items():
        if lo <= position <= hi:
            return label
    return "Other"


# ─────────────────────────────────────────────
# DURATION HELPERS
# ─────────────────────────────────────────────
def ms_to_min_sec(ms: float) -> str:
    """Convert milliseconds to MM:SS string."""
    if pd.isna(ms):
        return "N/A"
    total_seconds = int(ms / 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def duration_bucket(ms: float) -> str:
    """Categorise track into short / standard / long / extended."""
    if pd.isna(ms):
        return "Unknown"
    minutes = ms / 60_000
    if minutes < 2.5:
        return "Short (<2:30)"
    elif minutes < 3.5:
        return "Standard (2:30-3:30)"
    elif minutes < 4.5:
        return "Long (3:30-4:30)"
    else:
        return "Extended (>4:30)"


# ─────────────────────────────────────────────
# POPULARITY HELPERS
# ─────────────────────────────────────────────
def popularity_bucket(score: float) -> str:
    """Bin a raw popularity score into named tiers."""
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Viral (80-100)"
    elif score >= 60:
        return "Popular (60-79)"
    elif score >= 40:
        return "Moderate (40-59)"
    else:
        return "Niche (<40)"


# ─────────────────────────────────────────────
# METRIC FORMATTING
# ─────────────────────────────────────────────
def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator