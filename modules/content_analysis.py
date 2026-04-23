# modules/content_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE, RANK_GROUPS


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def explicit_summary(df: pd.DataFrame) -> dict:
    total    = len(df)
    explicit = df['is_explicit'].sum()
    clean    = total - explicit

    return {
        "total":       total,
        "explicit":    int(explicit),
        "clean":       int(clean),
        "explicit_pct": round(explicit / total * 100, 2) if total else 0,
        "clean_pct":    round(clean    / total * 100, 2) if total else 0,
    }


def explicit_by_rank_group(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('rank_group')
          .agg(
              total=('is_explicit', 'count'),
              explicit=('is_explicit', 'sum'),
          )
          .assign(explicit_pct=lambda x: x['explicit'] / x['total'] * 100)
          .reset_index()
    )


def explicit_avg_popularity(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('is_explicit')
          .agg(
              avg_popularity=('popularity', 'mean'),
              avg_position=('position', 'mean'),
              track_count=('song_id', 'count'),
          )
          .reset_index()
          .assign(label=lambda x: x['is_explicit'].map(
              {True: 'Explicit', False: 'Clean'}
          ))
    )


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_explicit_donut(df: pd.DataFrame) -> go.Figure:
    s = explicit_summary(df)

    fig = go.Figure(go.Pie(
        labels=['Explicit', 'Clean'],
        values=[s['explicit'], s['clean']],
        hole=0.55,
        marker_colors=[ATLANTIC_COLORS['explicit'], ATLANTIC_COLORS['clean']],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
    ))
    fig.update_layout(
        title="🔞 Explicit vs Clean Content Share",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        annotations=[dict(
            text=f"Explicit<br>{s['explicit_pct']}%",
            x=0.5, y=0.5, font_size=16,
            showarrow=False, font_color='white'
        )],
    )
    return fig


def plot_explicit_by_rank(df: pd.DataFrame) -> go.Figure:
    data  = explicit_by_rank_group(df)
    order = list(RANK_GROUPS.keys())
    data['rank_group'] = pd.Categorical(data['rank_group'],
                                         categories=order, ordered=True)
    data = data.sort_values('rank_group')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['rank_group'],
        y=data['explicit_pct'],
        name='Explicit %',
        marker_color=ATLANTIC_COLORS['explicit'],
        text=data['explicit_pct'].round(1).astype(str) + '%',
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        x=data['rank_group'],
        y=100 - data['explicit_pct'],
        name='Clean %',
        marker_color=ATLANTIC_COLORS['clean'],
    ))
    fig.update_layout(
        barmode='stack',
        title="🔞 Explicit Content Rate by Rank Group",
        yaxis_title="Percentage (%)",
        xaxis_title="Rank Group",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_explicit_vs_popularity(df: pd.DataFrame) -> go.Figure:
    data = explicit_avg_popularity(df)

    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['label']],
            y=[row['avg_popularity']],
            name=row['label'],
            marker_color=(ATLANTIC_COLORS['explicit']
                          if row['label'] == 'Explicit'
                          else ATLANTIC_COLORS['clean']),
            text=[f"{row['avg_popularity']:.1f}"],
            textposition='outside',
        ))
    fig.update_layout(
        title="📊 Avg Popularity — Explicit vs Clean",
        yaxis_title="Average Popularity Score",
        template=CHART_TEMPLATE,
        height=360,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_explicit_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: position bucket × month → % explicit."""
    df2 = df.copy()
    df2['month']    = df2['date'].dt.to_period('M').astype(str)
    df2['pos_band'] = pd.cut(df2['position'],
                              bins=[0,10,25,50],
                              labels=['Top 10','Top 11-25','Top 26-50'])

    pivot = (
        df2.groupby(['month','pos_band'])['is_explicit']
           .mean()
           .mul(100)
           .unstack('pos_band')
           .fillna(0)
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale='Reds',
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        colorbar=dict(title='Explicit %'),
    ))
    fig.update_layout(
        title="🗓️ Explicit Content % — Monthly × Position Band",
        xaxis_title="Position Band",
        yaxis_title="Month",
        template=CHART_TEMPLATE,
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig