# modules/duration_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE, ms_to_min_sec


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def duration_summary(df: pd.DataFrame) -> dict:
    dur = df['duration_ms'].dropna()
    return {
        "mean_ms":   round(dur.mean(),  0),
        "median_ms": round(dur.median(),0),
        "min_ms":    round(dur.min(),   0),
        "max_ms":    round(dur.max(),   0),
        "std_ms":    round(dur.std(),   0),
        "mean_fmt":   ms_to_min_sec(dur.mean()),
        "median_fmt": ms_to_min_sec(dur.median()),
    }


def duration_by_popularity(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=['duration_ms','popularity'])
          .groupby('popularity_bucket')
          .agg(
              avg_duration_min=('duration_min', 'mean'),
              track_count=('song_id', 'count'),
              avg_position=('position', 'mean'),
          )
          .reset_index()
    )


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_duration_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=['duration_ms']),
        x='duration_min',
        nbins=40,
        color='duration_bucket',
        title="⏱️ Track Duration Distribution",
        labels={'duration_min': 'Duration (minutes)'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template=CHART_TEMPLATE,
    )
    fig.add_vline(
        x=df['duration_min'].median(),
        line_dash='dash',
        line_color=ATLANTIC_COLORS['accent'],
        annotation_text=f"Median: {ms_to_min_sec(df['duration_ms'].median())}",
        annotation_font_color='white',
    )
    fig.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_duration_vs_popularity(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df.dropna(subset=['duration_ms','popularity']),
        x='duration_min',
        y='popularity',
        color='duration_bucket',
        size='popularity',
        hover_data={'song': True, 'artist': True, 'position': True},
        title="🎯 Duration vs Popularity",
        labels={
            'duration_min': 'Duration (minutes)',
            'popularity':   'Popularity Score',
        },
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template=CHART_TEMPLATE,
        opacity=0.7,
    )
    fig.update_layout(
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_duration_by_rank_group(df: pd.DataFrame) -> go.Figure:
    order = ['Top 10', 'Top 11-25', 'Top 26-50']
    data  = (
        df.dropna(subset=['duration_ms'])
          .groupby('rank_group')['duration_min']
          .mean()
          .reindex(order)
          .reset_index()
    )

    fig = px.bar(
        data,
        x='rank_group',
        y='duration_min',
        color='duration_min',
        color_continuous_scale='Blues',
        title="⏱️ Average Duration by Rank Group",
        labels={'duration_min': 'Avg Duration (min)', 'rank_group': 'Rank Group'},
        text=data['duration_min'].round(2).astype(str) + ' min',
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_duration_bucket_bar(df: pd.DataFrame) -> go.Figure:
    order  = ['Short (<2:30)','Standard (2:30-3:30)','Long (3:30-4:30)','Extended (>4:30)']
    counts = (
        df['duration_bucket']
          .value_counts()
          .reindex(order)
          .reset_index()
    )
    counts.columns = ['bucket', 'count']

    fig = px.bar(
        counts,
        x='bucket',
        y='count',
        color='bucket',
        title="📏 Track Duration Categories",
        labels={'bucket': 'Duration Category', 'count': 'Track Count'},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template=CHART_TEMPLATE,
        text='count',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=360,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig