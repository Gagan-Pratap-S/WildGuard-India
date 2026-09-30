"""
WildGuard India — Wildlife Conservation Analytics & Risk Dashboard

Analyses wildlife population trends in India using the Living Planet Index (LPI)
2024 public dataset and presents findings through an interactive Streamlit dashboard.
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# --- Page config ---

st.set_page_config(
    page_title="WildGuard India — Wildlife Conservation Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3d2e 0%, #145a3e 100%);
}
section[data-testid="stSidebar"] * { color: #e0f2e9 !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 22px 18px 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(16, 185, 129, 0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.15);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #065f46;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #4b5563;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Insight Card ── */
.insight-card {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #78350f;
}

/* ── Risk Card ── */
.risk-card {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #7f1d1d;
}

/* ── Action Card ── */
.action-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #1e3a5f;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #065f46;
    margin: 28px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #a7f3d0;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #065f46 0%, #0d9488 50%, #10b981 100%);
    color: white;
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 24px;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 8px;
    color: white !important;
}
.hero-banner p {
    font-size: 1rem;
    opacity: 0.92;
    margin: 0;
    color: #d1fae5;
}

/* ── Metric delta colours ── */
div[data-testid="stMetricDelta"] > div { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Consistent colour scheme for trend categories, used across all charts
TREND_COLORS = {
    "Declining": "#ef4444",
    "Increasing": "#22c55e",
    "Stable": "#3b82f6",
    "Uncertain": "#9ca3af",
}

ECOSYSTEM_COLORS = {
    "Terrestrial": "#22c55e",
    "Freshwater": "#3b82f6",
    "Marine": "#06b6d4",
}


# --- Data loading & cleaning ---

@st.cache_data(show_spinner="Loading & cleaning LPI data …")
def load_and_clean_data():
    """Load LPI dataset, filter to India, clean, return wide + long DataFrames."""
    data_path = os.path.join(os.path.dirname(__file__), "LPD_2024_public.csv")
    raw = pd.read_csv(data_path, low_memory=False)

    india = raw[raw["Country"] == "India"].copy()

    # Keep only non-replicate, quality-checked series selected for LPR2024
    india = india[india["Included in LPR2024"] == 1].copy()

    year_cols = [str(y) for y in range(1950, 2021)]
    for yc in year_cols:
        india[yc] = pd.to_numeric(india[yc], errors="coerce")

    empty_years = [y for y in year_cols if india[y].isna().all()]
    india.drop(columns=empty_years, inplace=True)
    year_cols = [y for y in year_cols if y not in empty_years]

    india.drop(columns=[c for c in india.columns if c.startswith("Unnamed")], inplace=True)

    cat_cols = [
        "Class", "Order", "Family", "Common_name", "Binomial",
        "System", "T_biome", "T_realm", "FW_realm", "FW_biome",
        "M_realm", "M_ocean", "M_biome", "Units", "Location", "Subspecies",
    ]
    for col in cat_cols:
        if col in india.columns:
            india[col] = india[col].replace("NULL", np.nan).str.strip()

    india["Latitude"] = pd.to_numeric(india["Latitude"], errors="coerce")
    india["Longitude"] = pd.to_numeric(india["Longitude"], errors="coerce")

    # Need at least 2 observations to compute a trend
    valid_mask = india[year_cols].notna().sum(axis=1) >= 2
    india = india[valid_mask].copy()

    meta_cols = [
        "ID", "Binomial", "Common_name", "Class", "Order", "Family",
        "Location", "Latitude", "Longitude",
        "System", "T_realm", "T_biome", "FW_realm", "FW_biome",
        "M_realm", "M_ocean", "M_biome", "Units",
    ]
    india = india[[c for c in meta_cols + year_cols if c in india.columns]].copy()
    india.reset_index(drop=True, inplace=True)

    # Long-form for time-series operations (aggregate index, etc.)
    long = india.melt(
        id_vars=meta_cols, value_vars=year_cols,
        var_name="Year", value_name="PopValue",
    )
    long["Year"] = long["Year"].astype(int)
    long = long.dropna(subset=["PopValue"])
    long = long[long["PopValue"] > 0].copy()
    long.reset_index(drop=True, inplace=True)

    return india, long, year_cols


# --- Trend calculations ---
# LPI series use different measurement units (counts, density, indices), so we
# compare relative change via log-linear regression rather than summing raw values.

@st.cache_data(show_spinner="Computing population trends …")
def compute_trends(_india, year_cols):
    """Compute per-series trend via log-linear regression: ln(pop) ~ year."""
    records = []
    for idx, row in _india.iterrows():
        vals = {int(y): row[y] for y in year_cols if pd.notna(row[y]) and row[y] > 0}
        if len(vals) < 2:
            continue

        years = np.array(sorted(vals.keys()))
        pops = np.array([vals[y] for y in years])
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, np.log(pops))

        annual_pct = (np.exp(slope) - 1) * 100
        span = years[-1] - years[0]
        total_change_pct = (np.exp(slope * span) - 1) * 100 if span > 0 else 0.0

        # Classify by slope magnitude and statistical significance
        if p_value < 0.1:
            if slope < -0.01:
                trend = "Declining"
            elif slope > 0.01:
                trend = "Increasing"
            else:
                trend = "Stable"
        else:
            trend = "Uncertain"

        records.append({
            "ID": row["ID"],
            "Binomial": row["Binomial"],
            "Common_name": row["Common_name"],
            "Class": row["Class"],
            "Order": row["Order"],
            "Family": row["Family"],
            "System": row["System"],
            "T_biome": row.get("T_biome", np.nan),
            "Units": row["Units"],
            "Latitude": row.get("Latitude", np.nan),
            "Longitude": row.get("Longitude", np.nan),
            "Location": row.get("Location", ""),
            "n_observations": len(vals),
            "first_year": int(years[0]),
            "last_year": int(years[-1]),
            "span_years": span,
            "first_value": pops[0],
            "last_value": pops[-1],
            "slope": slope,
            "annual_pct_change": annual_pct,
            "total_change_pct": total_change_pct,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "trend_class": trend,
        })

    return pd.DataFrame(records)


def compute_aggregate_index(long_df):
    """
    LPI-style aggregate index: average annual lambda = ln(N_t / N_{t-1})
    across all series, cumulated from a base of 100.
    """
    all_lambdas = []
    for sid, grp in long_df.groupby("ID"):
        grp = grp.sort_values("Year")
        years = grp["Year"].values
        vals = grp["PopValue"].values
        for i in range(1, len(years)):
            if vals[i] > 0 and vals[i - 1] > 0:
                all_lambdas.append({
                    "Year": years[i],
                    "lambda": np.log(vals[i] / vals[i - 1]),
                })

    if not all_lambdas:
        return pd.DataFrame(columns=["Year", "Index"])

    lam_df = pd.DataFrame(all_lambdas)
    mean_lam = lam_df.groupby("Year")["lambda"].mean().sort_index()

    idx_values = [100.0]
    years_sorted = mean_lam.index.tolist()
    for i in range(1, len(years_sorted)):
        idx_values.append(idx_values[-1] * np.exp(mean_lam.iloc[i]))

    return pd.DataFrame({"Year": years_sorted, "Index": idx_values})


# --- HTML card helpers ---

def kpi_card(value, label):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """

def insight_card(text):
    return f'<div class="insight-card">💡 {text}</div>'

def risk_card(text):
    return f'<div class="risk-card">⚠️ {text}</div>'

def action_card(text):
    return f'<div class="action-card">🎯 {text}</div>'


# --- Load data & compute trends ---

india, long, year_cols = load_and_clean_data()
trends = compute_trends(india, year_cols)
index_df = compute_aggregate_index(long)

# --- Global KPIs (unfiltered) ---

n_series = len(trends)
n_species = trends["Binomial"].nunique()
n_classes = trends["Class"].nunique()
n_declining = (trends["trend_class"] == "Declining").sum()
n_increasing = (trends["trend_class"] == "Increasing").sum()
pct_declining = round(n_declining / n_series * 100, 1) if n_series > 0 else 0
overall_median_change = round(trends["annual_pct_change"].median(), 2)
analysis_start = int(trends["first_year"].min())
analysis_end = int(trends["last_year"].max())

# --- Sidebar & filters ---

with st.sidebar:
    st.markdown("## 🐾 WildGuard India")
    st.markdown("Wildlife Conservation Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Executive Overview", "📈 Wildlife Trends", "🛡️ Conservation Risk & Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### Filters")

    all_classes = sorted(trends["Class"].dropna().unique())
    sel_classes = st.multiselect("Taxonomic Class", all_classes, default=all_classes)

    all_systems = sorted(trends["System"].dropna().unique())
    sel_systems = st.multiselect("Ecosystem", all_systems, default=all_systems)

    st.markdown("---")
    st.caption("Data: Living Planet Index 2024 (India)")
    st.caption("© WildGuard India Project")

filt = trends[
    trends["Class"].isin(sel_classes) &
    trends["System"].isin(sel_systems)
].copy()

f_n_series = len(filt)
f_n_species = filt["Binomial"].nunique()
f_n_declining = (filt["trend_class"] == "Declining").sum()
f_n_increasing = (filt["trend_class"] == "Increasing").sum()
f_pct_declining = round(f_n_declining / f_n_series * 100, 1) if f_n_series > 0 else 0
f_median_change = round(filt["annual_pct_change"].median(), 2) if f_n_series > 0 else 0


# =============================================================================
# Page 1 — Executive Overview
# =============================================================================

if page == "📊 Executive Overview":

    st.markdown("""
    <div class="hero-banner">
        <h1>🐾 WildGuard India</h1>
        <p>Wildlife Conservation Analytics & Risk Dashboard — powered by the Living Planet Index</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""This dashboard analyses **{n_series} population time-series** covering
        **{n_species} species** across **{n_classes} taxonomic classes** in India,
        drawn from the Living Planet Index 2024 public dataset.
        The data spans observations from **{analysis_start}** to **{analysis_end}**."""
    )

    st.markdown('<div class="section-header">Key Performance Indicators</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(kpi_card(f_n_series, "Population Series"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(f_n_species, "Unique Species"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(f_n_declining, "Declining Series"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card(f"{f_pct_declining}%", "% Declining"), unsafe_allow_html=True)
    with k5:
        trend_arrow = "↓" if f_median_change < 0 else "↑"
        st.markdown(kpi_card(f"{f_median_change}% {trend_arrow}", "Median Annual Δ"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card(n_classes, "Taxonomic Classes"), unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="section-header">Population Trend Overview</div>',
                unsafe_allow_html=True)

    col_trend, col_pie = st.columns([3, 2])

    with col_trend:
        if not index_df.empty:
            fig_idx = go.Figure()
            fig_idx.add_trace(go.Scatter(
                x=index_df["Year"], y=index_df["Index"],
                mode="lines+markers",
                line=dict(color="#059669", width=3),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(5,150,105,0.08)",
                name="LPI-style Index",
            ))
            fig_idx.add_hline(y=100, line_dash="dash", line_color="#9ca3af",
                              annotation_text="Baseline (100)")
            fig_idx.update_layout(
                title="Aggregate Wildlife Population Index (India)",
                xaxis_title="Year",
                yaxis_title="Index (base = 100)",
                template="plotly_white",
                height=380,
                margin=dict(l=50, r=30, t=50, b=40),
            )
            st.plotly_chart(fig_idx, width="stretch")
        else:
            st.info("Insufficient data to compute aggregate index.")

    with col_pie:
        trend_counts = filt["trend_class"].value_counts()
        fig_pie = px.pie(
            names=trend_counts.index,
            values=trend_counts.values,
            color=trend_counts.index,
            color_discrete_map=TREND_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(
            title="Trend Classification",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        fig_pie.update_traces(textinfo="label+percent", textfont_size=12)
        st.plotly_chart(fig_pie, width="stretch")

    # Auto-generated insights
    st.markdown('<div class="section-header">Automatically Derived Insights</div>',
                unsafe_allow_html=True)

    top_class = filt["Class"].value_counts().idxmax() if f_n_series > 0 else "N/A"
    top_class_n = filt["Class"].value_counts().max() if f_n_series > 0 else 0
    st.markdown(insight_card(
        f"**{top_class}** is the most monitored taxonomic class in the India LPI subset, "
        f"with **{top_class_n}** population series ({round(top_class_n/f_n_series*100, 1)}% of total)."
    ), unsafe_allow_html=True)

    top_sys = filt["System"].value_counts().idxmax() if f_n_series > 0 else "N/A"
    top_sys_n = filt["System"].value_counts().max() if f_n_series > 0 else 0
    st.markdown(insight_card(
        f"**{top_sys}** ecosystems account for the majority of monitored series "
        f"(**{top_sys_n}** series), suggesting monitoring effort is concentrated there."
    ), unsafe_allow_html=True)

    if f_n_declining > 0:
        worst = filt.loc[filt["annual_pct_change"].idxmin()]
        st.markdown(insight_card(
            f"**{f_n_declining} out of {f_n_series}** population series show a declining trend. "
            f"The steepest decline is observed in *{worst['Common_name']}* ({worst['Binomial']}), "
            f"with an estimated annual change of **{worst['annual_pct_change']:.1f}%** per year."
        ), unsafe_allow_html=True)


# =============================================================================
# Page 2 — Wildlife Trends
# =============================================================================

elif page == "📈 Wildlife Trends":

    st.markdown("""
    <div class="hero-banner">
        <h1>📈 Wildlife Trends</h1>
        <p>Population trends across taxonomic groups, ecosystems, and time</p>
    </div>
    """, unsafe_allow_html=True)

    # Trend by class
    st.markdown('<div class="section-header">Trend Distribution by Taxonomic Class</div>',
                unsafe_allow_html=True)

    class_trend = filt.groupby(["Class", "trend_class"]).size().reset_index(name="Count")
    fig_class = px.bar(
        class_trend, x="Class", y="Count", color="trend_class",
        color_discrete_map=TREND_COLORS,
        barmode="group",
        labels={"trend_class": "Trend"},
    )
    fig_class.update_layout(
        template="plotly_white", height=400,
        margin=dict(l=40, r=30, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_class, width="stretch")

    # Annual change histogram and ecosystem box plot (clip extreme outliers)
    q_low, q_high = filt["annual_pct_change"].quantile([0.02, 0.98])
    clipped = filt[
        (filt["annual_pct_change"] >= q_low) &
        (filt["annual_pct_change"] <= q_high)
    ]

    col_hist, col_box = st.columns(2)

    with col_hist:
        st.markdown('<div class="section-header">Annual % Change Distribution</div>',
                    unsafe_allow_html=True)
        fig_hist = px.histogram(
            clipped, x="annual_pct_change",
            nbins=40,
            color_discrete_sequence=["#059669"],
            labels={"annual_pct_change": "Annual % Change"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ef4444",
                           annotation_text="No change")
        fig_hist.update_layout(
            template="plotly_white", height=380,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_box:
        st.markdown('<div class="section-header">Annual % Change by Ecosystem</div>',
                    unsafe_allow_html=True)
        fig_box = px.box(
            clipped, x="System", y="annual_pct_change",
            color="System",
            color_discrete_map=ECOSYSTEM_COLORS,
            labels={"annual_pct_change": "Annual % Change"},
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_box.update_layout(
            template="plotly_white", height=380,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_box, width="stretch")

    # Top declining
    st.markdown('<div class="section-header">Top 15 Most Declining Population Series</div>',
                unsafe_allow_html=True)

    top_decl = filt.nsmallest(15, "annual_pct_change")[
        ["Common_name", "Binomial", "Class", "System", "annual_pct_change",
         "total_change_pct", "first_year", "last_year", "n_observations"]
    ].copy()
    top_decl.columns = [
        "Common Name", "Scientific Name", "Class", "Ecosystem",
        "Annual Δ (%)", "Total Δ (%)", "From", "To", "Obs."
    ]
    top_decl["Annual Δ (%)"] = top_decl["Annual Δ (%)"].round(1)
    top_decl["Total Δ (%)"] = top_decl["Total Δ (%)"].round(1)

    fig_decl = px.bar(
        top_decl.iloc[::-1], x="Annual Δ (%)", y="Common Name",
        orientation="h",
        color="Annual Δ (%)",
        color_continuous_scale=["#ef4444", "#fca5a5"],
        labels={"Annual Δ (%)": "Annual % Change"},
    )
    fig_decl.update_layout(
        template="plotly_white", height=480,
        margin=dict(l=180, r=30, t=20, b=40),
        coloraxis_showscale=False,
        yaxis_title="",
    )
    st.plotly_chart(fig_decl, width="stretch")

    # Top increasing
    st.markdown('<div class="section-header">Top 15 Most Increasing Population Series</div>',
                unsafe_allow_html=True)

    top_inc = filt.nlargest(15, "annual_pct_change")[
        ["Common_name", "Binomial", "Class", "System", "annual_pct_change",
         "total_change_pct", "first_year", "last_year", "n_observations"]
    ].copy()
    top_inc.columns = [
        "Common Name", "Scientific Name", "Class", "Ecosystem",
        "Annual Δ (%)", "Total Δ (%)", "From", "To", "Obs."
    ]
    top_inc["Annual Δ (%)"] = top_inc["Annual Δ (%)"].round(1)
    top_inc["Total Δ (%)"] = top_inc["Total Δ (%)"].round(1)

    fig_inc = px.bar(
        top_inc, x="Annual Δ (%)", y="Common Name",
        orientation="h",
        color="Annual Δ (%)",
        color_continuous_scale=["#bbf7d0", "#22c55e"],
        labels={"Annual Δ (%)": "Annual % Change"},
    )
    fig_inc.update_layout(
        template="plotly_white", height=480,
        margin=dict(l=180, r=30, t=20, b=40),
        coloraxis_showscale=False,
        yaxis_title="",
    )
    st.plotly_chart(fig_inc, width="stretch")

    # Species coverage by class
    st.markdown('<div class="section-header">Species Coverage by Taxonomic Class</div>',
                unsafe_allow_html=True)

    species_per_class = filt.groupby("Class")["Binomial"].nunique().reset_index()
    species_per_class.columns = ["Class", "Species Count"]
    species_per_class = species_per_class.sort_values("Species Count", ascending=True)

    fig_spc = px.bar(
        species_per_class, x="Species Count", y="Class",
        orientation="h",
        color_discrete_sequence=["#0d9488"],
    )
    fig_spc.update_layout(
        template="plotly_white", height=320,
        margin=dict(l=140, r=30, t=20, b=40),
        yaxis_title="",
    )
    st.plotly_chart(fig_spc, width="stretch")


# =============================================================================
# Page 3 — Conservation Risk & Insights
# =============================================================================

elif page == "🛡️ Conservation Risk & Insights":

    st.markdown("""
    <div class="hero-banner">
        <h1>🛡️ Conservation Risk & Insights</h1>
        <p>Evidence-based risk assessment and recommended conservation actions</p>
    </div>
    """, unsafe_allow_html=True)

    # High-risk series
    st.markdown('<div class="section-header">High-Risk Population Series</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Population series classified as **high risk** exhibit a strong negative "
        "annual trend (< −3% per year) with statistical evidence (p < 0.1). "
        "These represent populations that, if current trends continue, face "
        "substantial decline over the coming decades."
    )

    high_risk = filt[
        (filt["trend_class"] == "Declining") &
        (filt["annual_pct_change"] < -3)
    ].sort_values("annual_pct_change")

    if len(high_risk) > 0:
        hr_display = high_risk[[
            "Common_name", "Binomial", "Class", "System", "Location",
            "annual_pct_change", "total_change_pct", "first_year", "last_year",
            "n_observations"
        ]].copy()
        hr_display.columns = [
            "Common Name", "Scientific Name", "Class", "Ecosystem", "Location",
            "Annual Δ (%)", "Total Δ (%)", "From", "To", "Obs."
        ]
        hr_display["Annual Δ (%)"] = hr_display["Annual Δ (%)"].round(1)
        hr_display["Total Δ (%)"] = hr_display["Total Δ (%)"].round(1)
        st.dataframe(
            hr_display.reset_index(drop=True),
            width="stretch",
            height=min(400, 40 + len(hr_display) * 35),
        )
        st.caption(f"Showing {len(high_risk)} high-risk population series.")
    else:
        st.info("No population series meet the high-risk threshold with current filters.")

    # Risk by taxonomic group and ecosystem
    st.markdown('<div class="section-header">Risk Indicators by Taxonomic Class</div>',
                unsafe_allow_html=True)

    col_risk1, col_risk2 = st.columns(2)

    with col_risk1:
        class_summary = filt.groupby("Class").agg(
            total=("ID", "count"),
            declining=("trend_class", lambda x: (x == "Declining").sum()),
            median_change=("annual_pct_change", "median"),
        ).reset_index()
        class_summary["pct_declining"] = (
            class_summary["declining"] / class_summary["total"] * 100
        ).round(1)
        class_summary = class_summary.sort_values("pct_declining", ascending=False)

        fig_risk = px.bar(
            class_summary, x="Class", y="pct_declining",
            color="pct_declining",
            color_continuous_scale=["#fde68a", "#ef4444"],
            labels={"pct_declining": "% Declining"},
        )
        fig_risk.update_layout(
            title="% Declining Series by Class",
            template="plotly_white", height=380,
            margin=dict(l=40, r=30, t=50, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_risk, width="stretch")

    with col_risk2:
        sys_summary = filt.groupby("System").agg(
            total=("ID", "count"),
            declining=("trend_class", lambda x: (x == "Declining").sum()),
            median_change=("annual_pct_change", "median"),
        ).reset_index()
        sys_summary["pct_declining"] = (
            sys_summary["declining"] / sys_summary["total"] * 100
        ).round(1)

        fig_sys_risk = px.bar(
            sys_summary, x="System", y="pct_declining",
            color="pct_declining",
            color_continuous_scale=["#fde68a", "#ef4444"],
            labels={"pct_declining": "% Declining"},
        )
        fig_sys_risk.update_layout(
            title="% Declining Series by Ecosystem",
            template="plotly_white", height=380,
            margin=dict(l=40, r=30, t=50, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_sys_risk, width="stretch")

    # Monitoring map
    st.markdown('<div class="section-header">Geographic Distribution of Monitored Populations</div>',
                unsafe_allow_html=True)

    map_data = filt.dropna(subset=["Latitude", "Longitude"]).copy()
    if len(map_data) > 0:
        fig_map = px.scatter_map(
            map_data,
            lat="Latitude", lon="Longitude",
            color="trend_class",
            color_discrete_map=TREND_COLORS,
            hover_name="Common_name",
            hover_data={"Binomial": True, "Class": True, "annual_pct_change": ":.1f",
                        "Latitude": ":.2f", "Longitude": ":.2f", "trend_class": False},
            zoom=4,
            center={"lat": 22.5, "lon": 80},
            size_max=12,
            labels={"trend_class": "Trend"},
        )
        fig_map.update_layout(
            map_style="carto-positron",
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_map, width="stretch")

    # Key findings (FACT → INSIGHT → RISK → ACTION)
    st.markdown('<div class="section-header">Key Findings & Conservation Insights</div>',
                unsafe_allow_html=True)

    declining_by_class = filt[filt["trend_class"] == "Declining"].groupby("Class").size()
    most_declining_class = declining_by_class.idxmax() if len(declining_by_class) > 0 else None

    declining_by_sys = filt[filt["trend_class"] == "Declining"].groupby("System").size()
    most_declining_sys = declining_by_sys.idxmax() if len(declining_by_sys) > 0 else None

    if most_declining_class and f_n_declining > 0:
        n_in_class = declining_by_class[most_declining_class]
        st.markdown(risk_card(
            f"**FACT:** {f_n_declining} of {f_n_series} monitored population series "
            f"({f_pct_declining}%) show a declining trend.<br>"
            f"**INSIGHT:** The class *{most_declining_class}* accounts for the highest "
            f"number of declining series ({n_in_class}), suggesting this group faces "
            f"disproportionate population pressure.<br>"
            f"**RISK:** Continued declines in {most_declining_class} could lead to "
            f"local population collapses and reduced ecosystem function.<br>"
        ), unsafe_allow_html=True)
        st.markdown(action_card(
            f"**ACTION:** Prioritise habitat protection and monitoring programmes "
            f"for {most_declining_class} species in India, with emphasis on populations "
            f"showing the steepest declines."
        ), unsafe_allow_html=True)

    if most_declining_sys:
        n_in_sys = declining_by_sys[most_declining_sys]
        sys_total = filt[filt["System"] == most_declining_sys].shape[0]
        sys_pct = round(n_in_sys / sys_total * 100, 1) if sys_total > 0 else 0
        st.markdown(risk_card(
            f"**FACT:** Among ecosystems, *{most_declining_sys}* species show the "
            f"highest count of declining series ({n_in_sys} of {sys_total}, {sys_pct}%).<br>"
            f"**INSIGHT:** This pattern may be associated with habitat degradation, "
            f"pollution, or anthropogenic pressure specific to {most_declining_sys.lower()} "
            f"environments in India.<br>"
            f"**RISK:** {most_declining_sys} biodiversity loss can have cascading effects "
            f"on ecosystem services such as water purification, flood regulation, and "
            f"food security."
        ), unsafe_allow_html=True)
        st.markdown(action_card(
            f"**ACTION:** Strengthen {most_declining_sys.lower()} ecosystem conservation "
            f"programmes, including wetland/river/coastal protection policies and "
            f"pollution control in key habitats."
        ), unsafe_allow_html=True)

    if f_n_increasing > 0:
        top_success = filt.nlargest(3, "annual_pct_change")
        names = ", ".join(top_success["Common_name"].values[:3])
        st.markdown(insight_card(
            f"**FACT:** {f_n_increasing} population series show an increasing trend, "
            f"indicating conservation successes or population recovery.<br>"
            f"**EXAMPLES:** {names}.<br>"
            f"**INSIGHT:** These recoveries may be associated with protected area "
            f"management, anti-poaching efforts, or habitat restoration programmes.<br>"
            f"**OPPORTUNITY:** Study and replicate the conservation strategies that "
            f"coincide with these population recoveries for other at-risk species."
        ), unsafe_allow_html=True)

    sparse_series = filt[filt["n_observations"] <= 3]
    if len(sparse_series) > 0:
        st.markdown(insight_card(
            f"**FACT:** {len(sparse_series)} of {f_n_series} population series "
            f"({round(len(sparse_series)/f_n_series*100, 1)}%) have only 2–3 data points, "
            f"limiting trend reliability.<br>"
            f"**INSIGHT:** Sparse monitoring data creates uncertainty in trend estimates "
            f"and may mask true population trajectories.<br>"
            f"**OPPORTUNITY:** Investing in long-term, systematic wildlife monitoring "
            f"programmes would substantially improve the evidence base for conservation "
            f"decisions in India."
        ), unsafe_allow_html=True)

    recent = filt[filt["last_year"] >= 2010]
    old = filt[filt["last_year"] < 2000]
    if len(old) > 0:
        st.markdown(insight_card(
            f"**FACT:** {len(old)} population series have no observations after 2000, "
            f"while {len(recent)} series have data extending to 2010 or later.<br>"
            f"**INSIGHT:** Outdated monitoring data may not reflect current population "
            f"status, especially given rapid land-use changes in India over the past two "
            f"decades.<br>"
            f"**OPPORTUNITY:** Re-survey historically monitored populations to update "
            f"baseline data and assess whether earlier trends have continued, reversed, "
            f"or accelerated."
        ), unsafe_allow_html=True)

    # Recommended actions
    st.markdown('<div class="section-header">Recommended Conservation Actions</div>',
                unsafe_allow_html=True)

    actions = [
        "**Strengthen monitoring:** Expand long-term population monitoring for under-surveyed taxonomic groups and ecosystems.",
        "**Protect critical habitats:** Focus habitat protection on regions and ecosystems where the highest concentration of declining species is observed.",
        "**Targeted species programmes:** Develop species-specific recovery plans for populations showing the steepest declines.",
        "**Replicate successes:** Identify and scale conservation strategies associated with increasing population trends.",
        "**Update outdated surveys:** Re-survey populations with data gaps exceeding 10 years to update conservation status assessments.",
    ]
    for a in actions:
        st.markdown(action_card(a), unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="section-header">Important Limitations</div>',
                unsafe_allow_html=True)

    st.markdown("""
    - **Observational data:** The LPI dataset records observational population estimates; trends are associational, not causal.
    - **Sampling bias:** Coverage is uneven across taxa and geography — well-studied species (birds, large mammals) are over-represented.
    - **Measurement heterogeneity:** Population values use different units (counts, density, indices) across series; trends are computed using relative change to account for this.
    - **Temporal sparsity:** Many India series have few data points (median ≈ 3), which limits the precision of trend estimates.
    - **No threat attribution:** The dataset does not include direct threat or driver information; associations between trends and possible causes are inferred.
    """)


# --- Footer ---

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.82rem;'>"
    "WildGuard India — Wildlife Conservation Analytics Dashboard<br>"
    "Data Source: Living Planet Index 2024 Public Data "
    "(<a href='https://www.livingplanetindex.org/data_portal' target='_blank'>"
    "livingplanetindex.org</a>)<br>"
    "Built with Streamlit · Plotly · Pandas · SciPy"
    "</div>",
    unsafe_allow_html=True,
)
