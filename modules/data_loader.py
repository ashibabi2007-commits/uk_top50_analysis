# modules/data_loader.py

import pandas as pd
import numpy as np
import streamlit as st
from utils.helpers import (
    split_artists, normalize_artist_name,
    get_rank_group, duration_bucket, popularity_bucket
)


@st.cache_data(show_spinner="Loading & validating dataset…")
def load_and_validate(filepath: str) -> pd.DataFrame:
    """
    Load the UK Top 50 CSV, validate schema,
    engineer features and return a clean DataFrame.
    """
    df = pd.read_csv(filepath)

    # ── 1. Column standardisation ─────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    required_cols = [
        'date', 'position', 'song', 'artist',
        'popularity', 'duration_ms', 'album_type',
        'total_tracks', 'is_explicit'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # ── 2. Type coercions ─────────────────────────
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['position']     = pd.to_numeric(df['position'],   errors='coerce').astype('Int64')
    df['popularity']   = pd.to_numeric(df['popularity'], errors='coerce')
    df['duration_ms']  = pd.to_numeric(df['duration_ms'],errors='coerce')
    df['total_tracks'] = pd.to_numeric(df['total_tracks'],errors='coerce').astype('Int64')

    # Normalise boolean-like explicit flag
    df['is_explicit'] = df['is_explicit'].map(
        lambda x: True if str(x).strip().lower() in ('true','1','yes') else False
    )

    # ── 3. Drop rows with critical nulls ──────────
    df.dropna(subset=['date','position','song','artist'], inplace=True)
    df = df[(df['position'] >= 1) & (df['position'] <= 50)]

    # ── 4. Artist feature engineering ────────────
    df['artist_list']       = df['artist'].apply(split_artists)
    df['artist_count']      = df['artist_list'].apply(len)
    df['is_collaboration']  = df['artist_count'] > 1
    df['primary_artist']    = df['artist_list'].apply(
        lambda lst: lst[0] if lst else 'Unknown'
    )

    # ── 5. Rank group ─────────────────────────────
    df['rank_group'] = df['position'].apply(get_rank_group)

    # ── 6. Duration features ──────────────────────
    df['duration_min']    = df['duration_ms'] / 60_000
    df['duration_bucket'] = df['duration_ms'].apply(duration_bucket)

    # ── 7. Popularity bucket ──────────────────────
    df['popularity_bucket'] = df['popularity'].apply(popularity_bucket)

    # ── 8. Album type normalise ───────────────────
    df['album_type'] = df['album_type'].str.strip().str.title().fillna('Unknown')

    # ── 9. Unique song identifier ─────────────────
    df['song_id'] = df['song'].str.lower().str.strip() + '||' + df['primary_artist'].str.lower()

    df.reset_index(drop=True, inplace=True)
    return df


def get_date_range(df: pd.DataFrame):
    return df['date'].min(), df['date'].max()


def filter_dataframe(
    df: pd.DataFrame,
    start_date,
    end_date,
    selected_artists: list | None = None,
    collab_filter: str = "All",           # "All" | "Solo" | "Collaboration"
    album_type_filter: list | None = None,
    explicit_filter: str = "All",         # "All" | "Explicit" | "Clean"
) -> pd.DataFrame:

    mask = (df['date'] >= pd.Timestamp(start_date)) & \
           (df['date'] <= pd.Timestamp(end_date))

    if selected_artists:
        mask &= df['primary_artist'].isin(selected_artists)

    if collab_filter == "Solo":
        mask &= ~df['is_collaboration']
    elif collab_filter == "Collaboration":
        mask &= df['is_collaboration']

    if album_type_filter:
        mask &= df['album_type'].isin(album_type_filter)

    if explicit_filter == "Explicit":
        mask &= df['is_explicit']
    elif explicit_filter == "Clean":
        mask &= ~df['is_explicit']

    return df[mask].copy()