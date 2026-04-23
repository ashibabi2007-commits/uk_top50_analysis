# modules/album_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE, RANK_GROUPS


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def album_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('album_type')
          .agg(
              tracks=('song_id', 'count'),
              avg_position=('position', 'mean'),
              avg_popularity=('popularity', 'mean'),
              avg_total_tracks=('total_tracks', 'mean'),
          )
          .reset_index()
          .sort_values('tracks', ascending=False)
    )


def single_vs_album_ratio(df: pd.DataFrame) -> dict:
    counts    = df['album_type'].value_counts()
    total     = counts.sum()
    single_n  = counts.get('Single', 0)
    album_n   = counts.get('Album',  0)

    return {
        "single_count": int(single_n),
        "album_count":  int(album_n),
        "single_pct":   round(single_n / total * 100, 2) if total else 0,
        "album_pct":    round(album_n  / total * 100, 2) if total else 0,
        "ratio":        round(single_n / album_n, 2)     if album_n else 0,
    }


def album_size_vs_chart(df: pd.DataFrame) -> pd.DataFrame:
    """Bin total_tracks into size categories and compute avg position."""
    df2 = df.dropna(subset=['total_tracks']).copy()
    df2['album_size_cat'] = pd.cut(
        df2['total_tracks'],
        bins=[0, 1, 5, 12, 20, 999],
        labels=['1 track','2-5 tracks','6-12 tracks','13-20 tracks','20+ tracks'],
    )
    return (
        df2.groupby('album_size_cat', observed=True)
           .agg(
               track_count=('song_id', 'count'),
               avg_position=('position', 'mean'),
               avg_popularity=('popularity', 'mean'),
           )
           .reset_index()
    )


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_album_type_donut(df: pd.DataFrame) -> go.Figure:
    data = album_type_summary(df)
    colors = {
        'Single':    ATLANTIC_COLORS['single'],
        'Album':     ATLANTIC_COLORS['album'],
        'Compilation': ATLANTIC_COLORS['accent'],
        'Unknown':   '#888888',
    }

    fig = go.Figure(go.Pie(
        labels=data['album_type'],
        values=data['tracks'],
        hole=0.55,
        marker_colors=[colors.get(t, '#888') for t in data['album_type']],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
    ))
    fig.update_layout(
        title="💿 Album Type Distribution",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_album_type_by_rank(df: pd.DataFrame) -> go.Figure:
    order = list(RANK_GROUPS.keys())
    data  = (
        df.groupby(['rank_group', 'album_type'])
          .size()
          .reset_index(name='count')
    )
    data['rank_group'] = pd.Categorical(data['rank_group'],
                                         categories=order, ordered=True)
    data = data.sort_values('rank_group')

    fig = px.bar(
        data,
        x='rank_group',
        y='count',
        color='album_type',
        barmode='group',
        title="💿 Album Type by Rank Group",
        labels={'count': 'Track Count', 'rank_group': 'Rank Group'},
        color_discrete_map={
            'Single': ATLANTIC_COLORS['single'],
            'Album':  ATLANTIC_COLORS['album'],
        },
        template=CHART_TEMPLATE,
        text='count',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_album_size_chart(df: pd.DataFrame) -> go.Figure:
    data = album_size_vs_chart(df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['album_size_cat'].astype(str),
        y=data['track_count'],
        name='Track Count',
        marker_color=ATLANTIC_COLORS['album'],
        yaxis='y',
        text=data['track_count'],
        textposition='outside',
    ))
    fig.add_trace(go.Scatter(
        x=data['album_size_cat'].astype(str),
        y=data['avg_position'],
        name='Avg Chart Position',
        mode='lines+markers',
        marker=dict(color=ATLANTIC_COLORS['accent'], size=10),
        line=dict(color=ATLANTIC_COLORS['accent'], width=2),
        yaxis='y2',
    ))
    fig.update_layout(
        title="📀 Album Size vs Chart Presence",
        xaxis_title="Album Size (Total Tracks)",
        yaxis=dict(title='Track Count', showgrid=False),
        yaxis2=dict(
            title='Avg Chart Position (lower=better)',
            overlaying='y',
            side='right',
            autorange='reversed',
        ),
        template=CHART_TEMPLATE,
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_total_tracks_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=['total_tracks']),
        x='total_tracks',
        nbins=30,
        color='album_type',
        title="📊 Total Tracks Distribution by Album Type",
        labels={'total_tracks': 'Total Tracks in Release'},
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