# modules/artist_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def compute_artist_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Total chart appearances (rows) per primary artist."""
    return (
        df.groupby('primary_artist')
          .agg(
              appearances=('song_id', 'count'),
              avg_position=('position', 'mean'),
              avg_popularity=('popularity', 'mean'),
              unique_songs=('song_id', 'nunique'),
              top_rank=('position', 'min'),
          )
          .reset_index()
          .sort_values('appearances', ascending=False)
    )


def compute_unique_artists_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """How many unique primary artists appear each day."""
    result = (
        df.groupby('date')['primary_artist']
          .nunique()
          .reset_index()
    )
    result.columns = ['date', 'unique_artists']
    return result

def compute_artist_concentration_index(df: pd.DataFrame,
                                        top_n: int = 5) -> dict:
    """
    ACI = share of total appearances held by the top-N artists.
    Returns dict with index value and supporting stats.
    """
    appearances = compute_artist_appearances(df)
    total = appearances['appearances'].sum()
    top_share = appearances.head(top_n)['appearances'].sum()
    aci = (top_share / total) * 100 if total else 0

    return {
        "aci":            round(aci, 2),
        "top_n":          top_n,
        "top_n_share":    round(top_share / total * 100, 2) if total else 0,
        "total_entries":  int(total),
        "unique_artists": int(appearances.shape[0]),
        "diversity_score": round(appearances.shape[0] / total * 100, 4) if total else 0,
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_artist_leaderboard(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = compute_artist_appearances(df).head(top_n)

    fig = px.bar(
        data,
        x='appearances',
        y='primary_artist',
        orientation='h',
        color='avg_popularity',
        color_continuous_scale='Blues',
        text='appearances',
        hover_data={
            'avg_position':   ':.1f',
            'avg_popularity': ':.1f',
            'unique_songs':   True,
            'top_rank':       True,
        },
        title=f"🎤 Top {top_n} Artists by Chart Appearances",
        labels={
            'appearances':    'Total Chart Appearances',
            'primary_artist': 'Artist',
            'avg_popularity': 'Avg Popularity',
        },
        template=CHART_TEMPLATE,
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_colorbar_title="Avg Popularity",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    fig.update_traces(textposition='outside')
    return fig


def plot_unique_artists_trend(df: pd.DataFrame) -> go.Figure:
    daily = compute_unique_artists_per_day(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['unique_artists'],
        mode='lines+markers',
        line=dict(color=ATLANTIC_COLORS['primary'], width=2),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(10,49,97,0.2)',
        name='Unique Artists',
        hovertemplate='%{x|%d %b %Y}<br>Unique Artists: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title="📅 Unique Artists per Day",
        xaxis_title="Date",
        yaxis_title="Unique Artists",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_artist_dominance_treemap(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = compute_artist_appearances(df).head(top_n)

    fig = px.treemap(
        data,
        path=['primary_artist'],
        values='appearances',
        color='avg_popularity',
        color_continuous_scale='RdBu',
        title=f"🗺️ Artist Dominance Treemap (Top {top_n})",
        template=CHART_TEMPLATE,
        hover_data={'avg_position': ':.1f'},
    )
    fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig


def plot_avg_position_scatter(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = compute_artist_appearances(df).head(top_n)

    fig = px.scatter(
        data,
        x='appearances',
        y='avg_position',
        size='unique_songs',
        color='avg_popularity',
        color_continuous_scale='Viridis',
        text='primary_artist',
        title="🎯 Artist Appearances vs Average Chart Position",
        labels={
            'appearances':  'Chart Appearances',
            'avg_position': 'Avg Chart Position (lower = better)',
        },
        template=CHART_TEMPLATE,
        hover_data={'top_rank': True, 'unique_songs': True},
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig