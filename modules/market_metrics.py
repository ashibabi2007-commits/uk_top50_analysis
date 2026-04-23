# modules/market_metrics.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE, safe_divide


# ─────────────────────────────────────────────
# KPI COMPUTATIONS
# ─────────────────────────────────────────────
def compute_all_kpis(df: pd.DataFrame) -> dict:
    total_entries  = len(df)
    unique_artists = df['primary_artist'].nunique()
    unique_songs   = df['song_id'].nunique()

    # Artist Concentration Index (Top-5 share)
    top5_share = (
        df['primary_artist']
        .value_counts()
        .head(5)
        .sum()
    )
    aci = safe_divide(top5_share, total_entries) * 100

    # Diversity Score
    diversity_score = safe_divide(unique_artists, total_entries) * 100

    # Collaboration Ratio
    collab_ratio = safe_divide(df['is_collaboration'].sum(), total_entries) * 100

    # Explicit Share
    explicit_share = safe_divide(df['is_explicit'].sum(), total_entries) * 100

    # Single vs Album Ratio
    single_count = (df['album_type'] == 'Single').sum()
    album_count  = (df['album_type'] == 'Album').sum()
    single_ratio = safe_divide(single_count, total_entries) * 100

    # Content Variety Index = unique songs / total entries
    cvi = safe_divide(unique_songs, total_entries) * 100

    # Avg Popularity
    avg_popularity = df['popularity'].mean()

    # Avg Duration
    avg_duration_min = df['duration_min'].mean()

    return {
        "artist_concentration_index": round(aci, 2),
        "unique_artist_count":        int(unique_artists),
        "diversity_score":            round(diversity_score, 4),
        "collaboration_ratio":        round(collab_ratio, 2),
        "explicit_content_share":     round(explicit_share, 2),
        "single_ratio":               round(single_ratio, 2),
        "album_ratio":                round(safe_divide(album_count, total_entries) * 100, 2),
        "content_variety_index":      round(cvi, 2),
        "avg_popularity":             round(avg_popularity, 2),
        "avg_duration_min":           round(avg_duration_min, 2),
        "total_entries":              int(total_entries),
        "unique_songs":               int(unique_songs),
        "single_count":               int(single_count),
        "album_count":                int(album_count),
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_kpi_radar(kpis: dict) -> go.Figure:
    categories = [
        'Artist Concentration\nIndex',
        'Diversity Score',
        'Collaboration\nRatio',
        'Explicit Content\nShare',
        'Single Ratio',
        'Content Variety\nIndex',
    ]
    values = [
        kpis['artist_concentration_index'],
        kpis['diversity_score'] * 10,       # scale for visibility
        kpis['collaboration_ratio'],
        kpis['explicit_content_share'],
        kpis['single_ratio'],
        kpis['content_variety_index'],
    ]
    values += values[:1]   # close the polygon

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(10,49,97,0.4)',
        line=dict(color=ATLANTIC_COLORS['primary'], width=2),
        marker=dict(color=ATLANTIC_COLORS['accent'], size=6),
        name='UK Market Profile',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)',
        ),
        title="🕸️ UK Market Structure Radar",
        template=CHART_TEMPLATE,
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_popularity_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x='popularity',
        nbins=30,
        color='album_type',
        title="📊 Popularity Score Distribution by Album Type",
        labels={'popularity': 'Popularity Score'},
        color_discrete_map={
            'Single': ATLANTIC_COLORS['single'],
            'Album':  ATLANTIC_COLORS['album'],
        },
        template=CHART_TEMPLATE,
        barmode='overlay',
        opacity=0.75,
    )
    fig.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_position_over_time(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    top_artists = (
        df['primary_artist']
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    data = df[df['primary_artist'].isin(top_artists)].copy()
    daily_avg = (
        data.groupby(['date','primary_artist'])['position']
            .mean()
            .reset_index()
    )

    fig = px.line(
        daily_avg,
        x='date',
        y='position',
        color='primary_artist',
        title=f"📈 Chart Position Over Time — Top {top_n} Artists",
        labels={'position': 'Avg Chart Position', 'date': 'Date'},
        template=CHART_TEMPLATE,
        markers=True,
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig