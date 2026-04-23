# modules/collaboration_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from itertools import combinations
from utils.helpers import ATLANTIC_COLORS, CHART_TEMPLATE, RANK_GROUPS


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def collab_summary(df: pd.DataFrame) -> dict:
    total         = len(df)
    collab_tracks = df['is_collaboration'].sum()
    solo_tracks   = total - collab_tracks

    collab_pct = (collab_tracks / total * 100) if total else 0
    avg_collaborators = df[df['is_collaboration']]['artist_count'].mean()

    return {
        "total_tracks":    total,
        "solo_tracks":     int(solo_tracks),
        "collab_tracks":   int(collab_tracks),
        "collab_pct":      round(collab_pct, 2),
        "solo_pct":        round(100 - collab_pct, 2),
        "avg_collaborators": round(avg_collaborators, 2) if not np.isnan(avg_collaborators) else 0,
    }


def collab_by_rank_group(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('rank_group')
          .agg(
              total=('is_collaboration', 'count'),
              collaborations=('is_collaboration', 'sum'),
          )
          .assign(collab_pct=lambda x: x['collaborations'] / x['total'] * 100)
          .reset_index()
    )


def build_collaboration_network(df: pd.DataFrame,
                                 min_collabs: int = 2):
    """
    Build a NetworkX graph where nodes = artists,
    edges = how many times they appeared together.
    """
    G = nx.Graph()
    collab_df = df[df['is_collaboration']].copy()

    for _, row in collab_df.iterrows():
        artists = row['artist_list']
        for a, b in combinations(artists, 2):
            if G.has_edge(a, b):
                G[a][b]['weight'] += 1
            else:
                G.add_edge(a, b, weight=1)

    # Filter weak edges
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d['weight'] < min_collabs
    ]
    G.remove_edges_from(edges_to_remove)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_solo_vs_collab_donut(df: pd.DataFrame) -> go.Figure:
    summary = collab_summary(df)

    fig = go.Figure(go.Pie(
        labels=['Solo', 'Collaboration'],
        values=[summary['solo_tracks'], summary['collab_tracks']],
        hole=0.55,
        marker_colors=[ATLANTIC_COLORS['solo'], ATLANTIC_COLORS['collab']],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
    ))
    fig.update_layout(
        title="🎵 Solo vs Collaboration Tracks",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        annotations=[dict(
            text=f"Collab<br>{summary['collab_pct']}%",
            x=0.5, y=0.5, font_size=16, showarrow=False,
            font_color='white'
        )],
    )
    return fig


def plot_collab_by_rank(df: pd.DataFrame) -> go.Figure:
    data = collab_by_rank_group(df)
    order = list(RANK_GROUPS.keys())
    data['rank_group'] = pd.Categorical(data['rank_group'],
                                         categories=order, ordered=True)
    data = data.sort_values('rank_group')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['rank_group'],
        y=data['collab_pct'],
        name='Collaboration %',
        marker_color=ATLANTIC_COLORS['collab'],
        text=data['collab_pct'].round(1).astype(str) + '%',
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        x=data['rank_group'],
        y=100 - data['collab_pct'],
        name='Solo %',
        marker_color=ATLANTIC_COLORS['solo'],
    ))
    fig.update_layout(
        barmode='stack',
        title="🤝 Collaboration Rate by Rank Group",
        yaxis_title="Percentage (%)",
        xaxis_title="Rank Group",
        template=CHART_TEMPLATE,
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig


def plot_network_graph(G: nx.Graph) -> go.Figure:
    """Convert NetworkX graph to an interactive Plotly scatter plot."""
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No collaboration network data available",
            template=CHART_TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
        )
        return fig

    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Edges
    edge_x, edge_y, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(data['weight'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        showlegend=False,
    )

    # Nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    degree = dict(G.degree())
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Connections: {degree[node]}")
        node_size.append(10 + degree[node] * 5)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        textfont=dict(size=9, color='white'),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=[degree[n] for n in G.nodes()],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='Connections'),
            line=dict(width=1, color='white'),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="🕸️ Artist Collaboration Network",
        showlegend=False,
        hovermode='closest',
        template=CHART_TEMPLATE,
        height=520,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plot_collaborator_count_dist(df: pd.DataFrame) -> go.Figure:
    collab_df = df[df['is_collaboration']].copy()
    counts    = collab_df['artist_count'].value_counts().reset_index()
    counts.columns = ['collaborators', 'tracks']
    counts = counts.sort_values('collaborators')

    fig = px.bar(
        counts,
        x='collaborators',
        y='tracks',
        title="👥 Distribution of Collaborators per Track",
        labels={'collaborators': 'Number of Artists', 'tracks': 'Track Count'},
        color='tracks',
        color_continuous_scale='Purples',
        template=CHART_TEMPLATE,
        text='tracks',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    return fig