# app.py  ── United Kingdom Top 50 Playlist Analysis Dashboard
# Atlantic Recording Corporation | UK Market Intelligence

import streamlit as st
import pandas as pd
import os

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="UK Top 50 | Atlantic RC",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────────
from modules.data_loader import load_and_validate, filter_dataframe, get_date_range
from modules.artist_analysis import (
    compute_artist_appearances,
    compute_artist_concentration_index,
    plot_artist_leaderboard,
    plot_unique_artists_trend,
    plot_artist_dominance_treemap,
    plot_avg_position_scatter,
)
from modules.collaboration_analysis import (
    collab_summary,
    build_collaboration_network,
    plot_solo_vs_collab_donut,
    plot_collab_by_rank,
    plot_network_graph,
    plot_collaborator_count_dist,
)
from modules.content_analysis import (
    explicit_summary,
    explicit_avg_popularity,
    plot_explicit_donut,
    plot_explicit_by_rank,
    plot_explicit_vs_popularity,
    plot_explicit_heatmap,
)
from modules.album_analysis import (
    album_type_summary,
    single_vs_album_ratio,
    plot_album_type_donut,
    plot_album_type_by_rank,
    plot_album_size_chart,
    plot_total_tracks_distribution,
)
from modules.duration_analysis import (
    duration_summary,
    plot_duration_histogram,
    plot_duration_vs_popularity,
    plot_duration_by_rank_group,
    plot_duration_bucket_bar,
)
from modules.market_metrics import (
    compute_all_kpis,
    plot_kpi_radar,
    plot_popularity_distribution,
    plot_position_over_time,
)
from utils.helpers import fmt_pct, fmt_number, ms_to_min_sec, ATLANTIC_COLORS

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp { background-color: #0D1117; color: #E6EDF3; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A3161 0%, #0D1117 100%);
        border-right: 1px solid #21262D;
    }
    [data-testid="stSidebar"] * { color: #E6EDF3 !important; }

    /* ── KPI metric cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #161B22 0%, #21262D 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD700;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #8B949E;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 6px;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(90deg, #0A3161, #C8102E);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        margin: 20px 0 12px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ── Insight boxes ── */
    .insight-box {
        background: #161B22;
        border-left: 4px solid #FFD700;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #E6EDF3;
    }

    /* ── DataFrames ── */
    .dataframe thead { background-color: #0A3161 !important; }

    /* ── Hide default Streamlit header ── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPER: KPI card renderer
# ════════════════════════════════════════════════════════════════════════════
def kpi_card(label: str, value: str, delta: str = "", color: str = "#FFD700"):
    delta_html = (
        f'<div class="kpi-delta" style="color:{color};">{delta}</div>'
        if delta else ""
    )
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join("data", "uk_top50.csv")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. "
             "Please place your CSV file in the `data/` folder.")
    st.stop()

df_raw = load_and_validate(DATA_PATH)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — GLOBAL FILTERS
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem;'>🇬🇧</div>
        <div style='font-size:1.1rem; font-weight:700; color:#FFD700;'>
            UK Top 50 Analytics
        </div>
        <div style='font-size:0.75rem; color:#8B949E;'>
            Atlantic Recording Corporation
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    # ── Date range ──────────────────────────────────────────────────────────
    min_date, max_date = get_date_range(df_raw)
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    # ── Artist filter ────────────────────────────────────────────────────────
    all_artists = sorted(df_raw['primary_artist'].unique().tolist())
    selected_artists = st.multiselect(
        "🎤 Artist Filter",
        options=all_artists,
        default=[],
        placeholder="All artists",
    )

    # ── Collaboration toggle ─────────────────────────────────────────────────
    collab_filter = st.radio(
        "🤝 Track Type",
        options=["All", "Solo", "Collaboration"],
        horizontal=True,
    )

    # ── Album type filter ────────────────────────────────────────────────────
    album_types = sorted(df_raw['album_type'].unique().tolist())
    album_type_filter = st.multiselect(
        "💿 Album Type",
        options=album_types,
        default=album_types,
    )

    # ── Explicit filter ──────────────────────────────────────────────────────
    explicit_filter = st.radio(
        "🔞 Content Filter",
        options=["All", "Explicit", "Clean"],
        horizontal=True,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Display Options")
    top_n_artists  = st.slider("Top N Artists", 5, 30, 15)
    min_collabs    = st.slider("Min Collaborations (network)", 1, 10, 2)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#555;text-align:center;'>"
        "© Atlantic Recording Corporation<br>UK Market Intelligence v1.0"
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ════════════════════════════════════════════════════════════════════════════
df = filter_dataframe(
    df_raw,
    start_date      = start_date,
    end_date        = end_date,
    selected_artists= selected_artists if selected_artists else None,
    collab_filter   = collab_filter,
    album_type_filter= album_type_filter if album_type_filter else None,
    explicit_filter = explicit_filter,
)

if df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — NAVIGATION TABS
# ════════════════════════════════════════════════════════════════════════════
# Dashboard title
st.markdown("""
<div style='
    background: linear-gradient(135deg, #0A3161 0%, #C8102E 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
'>
    <h1 style='margin:0; color:white; font-size:2rem;'>
        🇬🇧 United Kingdom Top 50 Playlist Analysis
    </h1>
    <p style='margin:6px 0 0 0; color:#FFD700; font-size:1rem;'>
        Market Structure · Artist Diversity · Content Localisation
    </p>
    <p style='margin:4px 0 0 0; color:#ccc; font-size:0.8rem;'>
        Atlantic Recording Corporation — UK Strategic Intelligence Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# Tab navigation
tabs = st.tabs([
    "📊 Overview",
    "🎤 Artist Dominance",
    "🤝 Collaborations",
    "🔞 Content Analysis",
    "💿 Album Structure",
    "⏱️ Duration Insights",
    "📈 Market Metrics",
    "📋 Raw Data",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW (KPI Dashboard)
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    kpis = compute_all_kpis(df)

    # ── Row 1: Core KPIs ────────────────────────────────────────────────────
    section_header("🎯 Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Total Chart Entries",
            f"{kpis['total_entries']:,}",
            f"📅 {start_date} → {end_date}",
        )
    with c2:
        kpi_card(
            "Unique Artists",
            str(kpis['unique_artist_count']),
            f"Diversity Score: {kpis['diversity_score']}",
        )
    with c3:
        kpi_card(
            "Unique Songs",
            str(kpis['unique_songs']),
            f"CVI: {kpis['content_variety_index']}%",
        )
    with c4:
        kpi_card(
            "Avg Popularity Score",
            str(kpis['avg_popularity']),
            "0-100 scale",
            color="#27AE60",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card(
            "Artist Concentration Index",
            f"{kpis['artist_concentration_index']}%",
            "Top-5 artists' share",
            color="#E74C3C",
        )
    with c6:
        kpi_card(
            "Collaboration Ratio",
            f"{kpis['collaboration_ratio']}%",
            "Multi-artist tracks",
            color="#9B59B6",
        )
    with c7:
        kpi_card(
            "Explicit Content Share",
            f"{kpis['explicit_content_share']}%",
            "vs Clean content",
            color="#E74C3C",
        )
    with c8:
        kpi_card(
            "Single vs Album Ratio",
            f"{kpis['single_ratio']}%",
            f"Singles ({kpis['single_count']}) vs Albums ({kpis['album_count']})",
            color="#1ABC9C",
        )

    # ── Row 2: Quick-glance charts ──────────────────────────────────────────
    section_header("📊 Market Snapshot")
    col_a, col_b = st.columns(2)

    with col_a:
        st.plotly_chart(plot_kpi_radar(kpis), use_container_width=True)

    with col_b:
        st.plotly_chart(plot_popularity_distribution(df), use_container_width=True)

    # ── Insight summary ─────────────────────────────────────────────────────
    section_header("🔍 Executive Snapshot")

    aci  = kpis['artist_concentration_index']
    cratio = kpis['collaboration_ratio']
    ex   = kpis['explicit_content_share']
    sr   = kpis['single_ratio']

    insight_box(
        f"The top 5 artists account for {aci}% of chart entries — "
        f"{'high concentration' if aci > 30 else 'moderate concentration'} "
        f"indicating {'a star-driven' if aci > 30 else 'a diverse'} UK market."
    )
    insight_box(
        f"{cratio}% of charting tracks are collaborations, suggesting "
        f"{'strong' if cratio > 40 else 'moderate'} partnership culture in the UK Top 50."
    )
    insight_box(
        f"Explicit content represents {ex}% of the chart — "
        f"{'above' if ex > 50 else 'below'} the 50% threshold, "
        f"indicating UK listeners {'tolerate' if ex > 50 else 'prefer clean'} content."
    )
    insight_box(
        f"Singles dominate at {sr}% — UK labels favour single-format releases "
        f"for chart performance."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARTIST DOMINANCE
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("🎤 Artist Dominance & Diversity Analysis")

    # Concentration metrics
    aci_data = compute_artist_concentration_index(df, top_n=5)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Artist Concentration Index",
                 f"{aci_data['aci']}%", "Top-5 share")
    with m2:
        kpi_card("Unique Artists",
                 str(aci_data['unique_artists']), "In filtered period")
    with m3:
        kpi_card("Total Chart Entries",
                 f"{aci_data['total_entries']:,}", "Filtered entries")
    with m4:
        kpi_card("Diversity Score",
                 f"{aci_data['diversity_score']}", "Unique / Total × 100")

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard + treemap
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(
            plot_artist_leaderboard(df, top_n=top_n_artists),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_artist_dominance_treemap(df, top_n=top_n_artists),
            use_container_width=True,
        )

    # Unique artists trend
    st.plotly_chart(plot_unique_artists_trend(df), use_container_width=True)

    # Scatter: appearances vs avg position
    st.plotly_chart(plot_avg_position_scatter(df, top_n=top_n_artists),
                    use_container_width=True)

    # Leaderboard table
    section_header("📋 Artist Leaderboard Table")
    leaderboard = compute_artist_appearances(df).head(top_n_artists).copy()
    leaderboard['avg_position']   = leaderboard['avg_position'].round(1)
    leaderboard['avg_popularity'] = leaderboard['avg_popularity'].round(1)
    leaderboard.index = range(1, len(leaderboard) + 1)
    st.dataframe(
        leaderboard.rename(columns={
            'primary_artist': 'Artist',
            'appearances':    'Chart Appearances',
            'avg_position':   'Avg Position',
            'avg_popularity': 'Avg Popularity',
            'unique_songs':   'Unique Songs',
            'top_rank':       'Best Rank',
        }),
        use_container_width=True,
        height=400,
    )

    # Insights
    section_header("💡 Artist Dominance Insights")
    top_artist = compute_artist_appearances(df).iloc[0]
    insight_box(
        f"'{top_artist['primary_artist']}' leads the UK chart with "
        f"{top_artist['appearances']} appearances and an average position of "
        f"{top_artist['avg_position']:.1f}."
    )
    insight_box(
        f"Artist Concentration Index of {aci_data['aci']}% suggests the UK Top 50 "
        f"is {'highly concentrated' if aci_data['aci'] > 35 else 'moderately diverse'}, "
        f"with {'few artists dominating' if aci_data['aci'] > 35 else 'healthy competition'}."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLLABORATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("🤝 Collaboration Structure Analysis")

    summary = collab_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Collaboration Tracks",
                 str(summary['collab_tracks']),
                 f"{summary['collab_pct']}% of chart")
    with m2:
        kpi_card("Solo Tracks",
                 str(summary['solo_tracks']),
                 f"{summary['solo_pct']}% of chart")
    with m3:
        kpi_card("Avg Collaborators",
                 str(summary['avg_collaborators']),
                 "Per collaboration track")
    with m4:
        kpi_card("Collaboration Ratio",
                 f"{summary['collab_pct']}%", "vs Solo tracks")

    st.markdown("<br>", unsafe_allow_html=True)

    # Donut + rank group bar
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_solo_vs_collab_donut(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_collab_by_rank(df), use_container_width=True)

    # Collaborator count distribution
    st.plotly_chart(plot_collaborator_count_dist(df), use_container_width=True)

    # Network graph
    section_header("🕸️ Artist Collaboration Network")
    with st.spinner("Building collaboration network…"):
        G = build_collaboration_network(df, min_collabs=min_collabs)

    st.info(
        f"Network: **{G.number_of_nodes()} artists**, "
        f"**{G.number_of_edges()} collaboration pairs** "
        f"(min {min_collabs} co-appearances)"
    )
    st.plotly_chart(plot_network_graph(G), use_container_width=True)

    # Insights
    section_header("💡 Collaboration Insights")
    insight_box(
        f"With a {summary['collab_pct']}% collaboration rate, the UK Top 50 "
        f"shows {'heavy' if summary['collab_pct'] > 50 else 'moderate'} reliance "
        f"on multi-artist releases to drive chart performance."
    )
    insight_box(
        f"Average {summary['avg_collaborators']} artists per collaboration track "
        f"suggests {'duets dominate' if summary['avg_collaborators'] < 2.5 else 'group collaborations are common'}."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTENT (EXPLICIT) ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("🔞 Explicit Content Analysis — UK Cultural Sensitivity")

    ex_sum = explicit_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Explicit Tracks",
                 str(ex_sum['explicit']),
                 f"{ex_sum['explicit_pct']}% of chart",
                 color="#E74C3C")
    with m2:
        kpi_card("Clean Tracks",
                 str(ex_sum['clean']),
                 f"{ex_sum['clean_pct']}% of chart",
                 color="#27AE60")
    with m3:
        exp_pop = explicit_avg_popularity(df)
        exp_val = exp_pop[exp_pop['label'] == 'Explicit']['avg_popularity']
        kpi_card("Avg Explicit Popularity",
                 f"{exp_val.values[0]:.1f}" if len(exp_val) else "N/A",
                 "0-100 scale", color="#E74C3C")
    with m4:
        cln_val = exp_pop[exp_pop['label'] == 'Clean']['avg_popularity']
        kpi_card("Avg Clean Popularity",
                 f"{cln_val.values[0]:.1f}" if len(cln_val) else "N/A",
                 "0-100 scale", color="#27AE60")

    st.markdown("<br>", unsafe_allow_html=True)

    # Donut + rank group
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_explicit_donut(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_explicit_by_rank(df), use_container_width=True)

    # Popularity comparison
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_explicit_vs_popularity(df), use_container_width=True)
    with col4:
        st.plotly_chart(plot_explicit_heatmap(df), use_container_width=True)

    # Insights
    section_header("💡 Content Sensitivity Insights")
    insight_box(
        f"Explicit content holds {ex_sum['explicit_pct']}% of UK chart share — "
        f"{'notable prevalence' if ex_sum['explicit_pct'] > 50 else 'minority presence'} "
        f"suggesting UK listeners {'engage heavily with' if ex_sum['explicit_pct'] > 50 else 'marginally tolerate'} explicit material."
    )
    insight_box(
        "Atlantic RC should consider UK content localisation: "
        "clean edits alongside explicit releases may improve radio airplay and playlist inclusion."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALBUM STRUCTURE
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("💿 Album Structure & Release Strategy Analysis")

    ratio = single_vs_album_ratio(df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Single Tracks",
                 str(ratio['single_count']),
                 f"{ratio['single_pct']}% of chart",
                 color=ATLANTIC_COLORS['single'])
    with m2:
        kpi_card("Album Tracks",
                 str(ratio['album_count']),
                 f"{ratio['album_pct']}% of chart",
                 color=ATLANTIC_COLORS['album'])
    with m3:
        kpi_card("Single:Album Ratio",
                 f"{ratio['ratio']}:1",
                 "Singles dominate" if ratio['ratio'] > 1 else "Albums dominate",
                 color=ATLANTIC_COLORS['accent'])
    with m4:
        alb_summary = album_type_summary(df)
        avg_tracks  = alb_summary[alb_summary['album_type'] == 'Album']['avg_total_tracks']
        kpi_card("Avg Album Size",
                 f"{avg_tracks.values[0]:.0f} tracks" if len(avg_tracks) else "N/A",
                 "For Album-type releases")

    st.markdown("<br>", unsafe_allow_html=True)

    # Donut + rank group
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_album_type_donut(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_album_type_by_rank(df), use_container_width=True)

    # Album size charts
    st.plotly_chart(plot_album_size_chart(df), use_container_width=True)
    st.plotly_chart(plot_total_tracks_distribution(df), use_container_width=True)

    # Insights
    section_header("💡 Release Strategy Insights")
    insight_box(
        f"Singles represent {ratio['single_pct']}% of UK chart entries — "
        f"Atlantic RC should prioritise single releases for UK chart penetration."
    )
    insight_box(
        "Album-length projects with targeted singles can extend chart longevity. "
        "Consider staggered single drops ahead of album releases in the UK market."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DURATION INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("⏱️ Track Duration & Format Analysis")

    dur_sum = duration_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Mean Duration",
                 dur_sum['mean_fmt'], f"{dur_sum['mean_ms']/60000:.2f} min")
    with m2:
        kpi_card("Median Duration",
                 dur_sum['median_fmt'], f"{dur_sum['median_ms']/60000:.2f} min")
    with m3:
        kpi_card("Shortest Track",
                 ms_to_min_sec(dur_sum['min_ms']), "Minimum duration")
    with m4:
        kpi_card("Longest Track",
                 ms_to_min_sec(dur_sum['max_ms']), "Maximum duration")

    st.markdown("<br>", unsafe_allow_html=True)

    # Histogram + bucket bar
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_duration_histogram(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_duration_bucket_bar(df), use_container_width=True)

    # Duration vs popularity + by rank group
    st.plotly_chart(plot_duration_vs_popularity(df), use_container_width=True)
    st.plotly_chart(plot_duration_by_rank_group(df), use_container_width=True)

    # Insights
    section_header("💡 Duration Insights")
    insight_box(
        f"The median UK Top 50 track runs {dur_sum['median_fmt']} — "
        "reflecting standard radio-friendly format preferences."
    )
    insight_box(
        "Tracks in the 2:30–3:30 range typically align with streaming skip behaviour. "
        "Atlantic RC artists should target this window for UK radio and playlist optimisation."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET METRICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section_header("📈 Market Structure Metrics")

    kpis = compute_all_kpis(df)

    # Full KPI table
    kpi_df = pd.DataFrame([
        {"KPI": "Artist Concentration Index (ACI)",
         "Value": f"{kpis['artist_concentration_index']}%",
         "Description": "Share of top-5 artists in total chart entries"},
        {"KPI": "Unique Artist Count",
         "Value": str(kpis['unique_artist_count']),
         "Description": "Number of distinct primary artists in filtered data"},
        {"KPI": "Diversity Score",
         "Value": str(kpis['diversity_score']),
         "Description": "Unique artists / total entries × 100"},
        {"KPI": "Collaboration Ratio",
         "Value": f"{kpis['collaboration_ratio']}%",
         "Description": "Proportion of multi-artist tracks"},
        {"KPI": "Explicit Content Share",
         "Value": f"{kpis['explicit_content_share']}%",
         "Description": "Proportion of explicit tracks"},
        {"KPI": "Single Ratio",
         "Value": f"{kpis['single_ratio']}%",
         "Description": "Proportion of single-type releases"},
        {"KPI": "Album Ratio",
         "Value": f"{kpis['album_ratio']}%",
         "Description": "Proportion of album-type releases"},
        {"KPI": "Content Variety Index (CVI)",
         "Value": f"{kpis['content_variety_index']}%",
         "Description": "Unique songs / total entries × 100"},
        {"KPI": "Average Popularity Score",
         "Value": str(kpis['avg_popularity']),
         "Description": "Mean popularity across filtered tracks"},
        {"KPI": "Average Duration",
         "Value": f"{kpis['avg_duration_min']:.2f} min",
         "Description": "Mean track duration in minutes"},
    ])

    st.dataframe(kpi_df, use_container_width=True, hide_index=True, height=380)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar + position over time
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_kpi_radar(kpis), use_container_width=True)
    with col2:
        st.plotly_chart(plot_position_over_time(df, top_n=5),
                        use_container_width=True)

    # Insights
    section_header("💡 Strategic Market Insights")
    insight_box(
        "A high Diversity Score combined with a high ACI indicates a two-tier market: "
        "a small group of dominant artists alongside a long tail of niche performers."
    )
    insight_box(
        "The Content Variety Index reveals how frequently new/unique songs enter the chart — "
        "a high CVI suggests rapid song turnover, benefiting artists who release frequently."
    )
    insight_box(
        "Atlantic RC recommendation: invest in both star-anchored releases (ACI strategy) "
        "and long-tail discovery playlists (Diversity strategy) for comprehensive UK coverage."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — RAW DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section_header("📋 Filtered Dataset Explorer")

    st.info(
        f"Showing **{len(df):,}** records | "
        f"Date range: **{start_date}** → **{end_date}** | "
        f"Unique artists: **{df['primary_artist'].nunique()}**"
    )

    # Column selector
    display_cols = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=[
            'date', 'position', 'song', 'artist',
            'popularity', 'duration_min', 'album_type',
            'total_tracks', 'is_explicit', 'is_collaboration',
            'rank_group', 'duration_bucket',
        ],
    )

    # Search box
    search_term = st.text_input("🔍 Search by song or artist", "")
    display_df  = df[display_cols].copy() if display_cols else df.copy()

    if search_term:
        mask = (
            df['song'].str.lower().str.contains(search_term.lower(), na=False) |
            df['artist'].str.lower().str.contains(search_term.lower(), na=False)
        )
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, height=500)

    # Download
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_data,
        file_name=f"uk_top50_filtered_{start_date}_{end_date}.csv",
        mime="text/csv",
    )