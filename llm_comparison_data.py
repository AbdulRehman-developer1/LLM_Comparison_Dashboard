"""
streamlit_testing.py
=====================
Advanced, dynamic Streamlit dashboard for exploring an LLM comparison
dataset. Features a full control sidebar, KPI cards, and a set of
advanced/professional Plotly visuals (bubble scatter, treemap, radar,
parallel coordinates, correlation heatmap, leaderboard curve, box plots)
that are all driven by the same filtered dataframe -- so every chart
reacts instantly to the sidebar filters.

Expected project layout:
    project/
    ├── llm_comparison_data.py
    └── dataset/
        └── llm_comparison_dataset.csv   (or any *.csv with similar columns)

Run with:
    streamlit run llm_comparison.py
"""

import glob
import os

import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM CSS - professional look & feel
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1c2333 0%, #161b29 100%);
            border: 1px solid #2a3346;
            border-radius: 14px;
            padding: 14px 16px 8px 16px;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.8; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #2a3346;
        }
        h1, h2, h3 { letter-spacing: 0.3px; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# DATA LOADING
# Looks in ./dataset/*.csv first (matches the user's project layout),
# falls back to any *.csv nearby, and finally lets the user upload one.
# ----------------------------------------------------------------------
REQUIRED_COLS = {
    "Model", "Provider", "Context Window", "Speed (tokens/sec)", "Latency (sec)",
    "Benchmark (MMLU)", "Benchmark (Chatbot Arena)", "Open-Source",
    "Price / Million Tokens", "Training Dataset Size", "Compute Power",
    "Energy Efficiency", "Quality Rating", "Speed Rating", "Price Rating",
}


def _find_dataset_path():
    candidates = []
    for pattern in ("dataset/*.csv", "data/*.csv", "*.csv"):
        candidates.extend(glob.glob(pattern))
    for c in candidates:
        if "llm" in os.path.basename(c).lower():
            return c
    return candidates[0] if candidates else None


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    data = pd.read_csv(path)
    data["Open-Source"] = data["Open-Source"].astype(int)
    return data


dataset_path = _find_dataset_path()
df_raw = None
if dataset_path is not None:
    try:
        df_raw = load_data(dataset_path)
        if not REQUIRED_COLS.issubset(set(df_raw.columns)):
            df_raw = None
    except Exception:
        df_raw = None

if df_raw is None:
    st.sidebar.warning("Couldn't auto-find the dataset in `./dataset/`.")
    uploaded = st.sidebar.file_uploader("Upload the LLM comparison CSV", type="csv")
    if uploaded is None:
        st.title("🤖 LLM Comparison Dashboard")
        st.info(
            "Place your CSV inside a **dataset/** folder next to this script "
            "(e.g. `dataset/llm_comparison_dataset.csv`), or upload it using "
            "the control in the sidebar to get started."
        )
        st.stop()
    df_raw = pd.read_csv(uploaded)
    df_raw["Open-Source"] = df_raw["Open-Source"].astype(int)

# ----------------------------------------------------------------------
# CONSTANTS / HELPERS
# ----------------------------------------------------------------------
PROVIDERS = sorted(df_raw["Provider"].unique())
COLOR_SEQUENCE = px.colors.qualitative.Set2
PROVIDER_COLORS = {p: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, p in enumerate(PROVIDERS)}

NUMERIC_COLS = [
    "Context Window", "Speed (tokens/sec)", "Latency (sec)", "Benchmark (MMLU)",
    "Benchmark (Chatbot Arena)", "Price / Million Tokens", "Training Dataset Size",
    "Compute Power", "Energy Efficiency", "Quality Rating", "Speed Rating", "Price Rating",
]

# ----------------------------------------------------------------------
# SIDEBAR - control panel
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.caption(f"Source: `{dataset_path or 'uploaded file'}`  ·  {len(df_raw)} models")

    with st.expander("🎨 Display settings", expanded=False):
        plot_theme = st.radio("Chart theme", ["Dark", "Light"], horizontal=True, index=0)
        show_labels = st.checkbox("Show model name labels on scatter plots", value=False)
    template = "plotly_dark" if plot_theme == "Dark" else "plotly_white"

    st.markdown("### 🔎 Filters")

    provider_filter = st.multiselect("Provider", PROVIDERS, default=PROVIDERS)

    source_filter = st.radio(
        "License type", ["All", "Open-Source only", "Closed-Source only"], index=0
    )

    search_term = st.text_input("Search model name", "")

    cw_min, cw_max = int(df_raw["Context Window"].min()), int(df_raw["Context Window"].max())
    context_range = st.slider(
        "Context Window (tokens)", cw_min, cw_max, (cw_min, cw_max), step=10000
    )

    price_min, price_max = float(df_raw["Price / Million Tokens"].min()), float(
        df_raw["Price / Million Tokens"].max()
    )
    price_range = st.slider(
        "Price / Million Tokens ($)", price_min, price_max, (price_min, price_max)
    )

    mmlu_min, mmlu_max = int(df_raw["Benchmark (MMLU)"].min()), int(df_raw["Benchmark (MMLU)"].max())
    mmlu_range = st.slider("Benchmark (MMLU)", mmlu_min, mmlu_max, (mmlu_min, mmlu_max))

    arena_min, arena_max = int(df_raw["Benchmark (Chatbot Arena)"].min()), int(
        df_raw["Benchmark (Chatbot Arena)"].max()
    )
    arena_range = st.slider("Chatbot Arena Elo", arena_min, arena_max, (arena_min, arena_max))

    latency_min, latency_max = float(df_raw["Latency (sec)"].min()), float(df_raw["Latency (sec)"].max())
    latency_range = st.slider("Latency (sec)", latency_min, latency_max, (latency_min, latency_max))

    quality_filter = st.multiselect(
        "Quality Rating", sorted(df_raw["Quality Rating"].unique()),
        default=sorted(df_raw["Quality Rating"].unique()),
    )

    st.markdown("### 📊 Chart controls")
    top_n = st.slider("Top-N models (leaderboard charts)", 5, 30, 10)
    sort_metric = st.selectbox(
        "Rank models by",
        ["Benchmark (MMLU)", "Benchmark (Chatbot Arena)", "Speed (tokens/sec)",
         "Energy Efficiency", "Price / Million Tokens"],
        index=0,
    )

    if st.button("♻️ Reset all filters", use_container_width=True):
        st.rerun()

# ----------------------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------------------
df = df_raw.copy()
if provider_filter:
    df = df[df["Provider"].isin(provider_filter)]
else:
    df = df.iloc[0:0]

if source_filter == "Open-Source only":
    df = df[df["Open-Source"] == 1]
elif source_filter == "Closed-Source only":
    df = df[df["Open-Source"] == 0]

if search_term.strip():
    df = df[df["Model"].str.contains(search_term.strip(), case=False, na=False)]

df = df[
    (df["Context Window"].between(*context_range))
    & (df["Price / Million Tokens"].between(*price_range))
    & (df["Benchmark (MMLU)"].between(*mmlu_range))
    & (df["Benchmark (Chatbot Arena)"].between(*arena_range))
    & (df["Latency (sec)"].between(*latency_range))
    & (df["Quality Rating"].isin(quality_filter))
]

st.title("🤖 LLM Comparison Dashboard")
st.caption(
    "Every chart below reads from the same filtered dataset — adjust the "
    "controls in the sidebar and watch the whole dashboard update together."
)

if df.empty:
    st.warning("No models match the current filter combination. Try loosening a filter.")
    st.stop()

# ----------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Models in view", len(df), delta=f"{len(df) - len(df_raw)} vs total")
k2.metric("Avg MMLU", f"{df['Benchmark (MMLU)'].mean():.1f}",
          delta=f"{df['Benchmark (MMLU)'].mean() - df_raw['Benchmark (MMLU)'].mean():+.1f}")
k3.metric("Avg Arena Elo", f"{df['Benchmark (Chatbot Arena)'].mean():.0f}",
          delta=f"{df['Benchmark (Chatbot Arena)'].mean() - df_raw['Benchmark (Chatbot Arena)'].mean():+.0f}")
k4.metric("Avg Price /M tok", f"${df['Price / Million Tokens'].mean():.2f}",
          delta=f"{df['Price / Million Tokens'].mean() - df_raw['Price / Million Tokens'].mean():+.2f}",
          delta_color="inverse")
k5.metric("Avg Latency", f"{df['Latency (sec)'].mean():.2f}s",
          delta=f"{df['Latency (sec)'].mean() - df_raw['Latency (sec)'].mean():+.2f}s",
          delta_color="inverse")
k6.metric("Open-Source %", f"{df['Open-Source'].mean() * 100:.0f}%")

st.divider()

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_overview, tab_perf, tab_price, tab_compare, tab_corr, tab_data = st.tabs(
    ["🌐 Overview", "🏆 Performance", "💰 Pricing & Efficiency",
     "🧭 Compare Models", "🔗 Correlations", "📄 Data Explorer"]
)

# ================= OVERVIEW =================
with tab_overview:
    c1, c2 = st.columns((3, 2))
    with c1:
        st.subheader("Quality vs Price — Bubble Map")
        fig = px.scatter(
            df, x="Price / Million Tokens", y="Benchmark (MMLU)",
            size="Speed (tokens/sec)", color="Provider",
            color_discrete_map=PROVIDER_COLORS,
            hover_name="Model",
            hover_data={"Latency (sec)": True, "Benchmark (Chatbot Arena)": True},
            text="Model" if show_labels else None,
            size_max=40, template=template,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=460, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Model Count by Provider")
        counts = df["Provider"].value_counts().reset_index()
        counts.columns = ["Provider", "Count"]
        fig = px.bar(
            counts.sort_values("Count"), x="Count", y="Provider", orientation="h",
            color="Provider", color_discrete_map=PROVIDER_COLORS, template=template,
        )
        fig.update_layout(height=460, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Provider → Model Treemap (sized by Training Data, colored by MMLU)")
    fig = px.treemap(
        df, path=["Provider", "Model"], values="Training Dataset Size",
        color="Benchmark (MMLU)", color_continuous_scale="RdYlGn",
        template=template,
    )
    fig.update_layout(height=480, margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ================= PERFORMANCE =================
with tab_perf:
    st.subheader(f"Top {top_n} Models — ranked by {sort_metric}")
    ranked = df.sort_values(sort_metric, ascending=False).head(top_n)
    fig = px.bar(
        ranked.sort_values(sort_metric), x=sort_metric, y="Model", orientation="h",
        color="Provider", color_discrete_map=PROVIDER_COLORS,
        hover_data=["Benchmark (MMLU)", "Benchmark (Chatbot Arena)", "Price / Million Tokens"],
        template=template,
    )
    fig.update_layout(height=max(350, 26 * top_n))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Leaderboard Curve")
        curve = df.sort_values(sort_metric, ascending=False).reset_index(drop=True)
        curve["Rank"] = curve.index + 1
        fig = px.line(
            curve, x="Rank", y=sort_metric, markers=True, color="Open-Source",
            hover_data=["Model", "Provider"], template=template,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Latency Spread by Provider")
        fig = px.box(
            df, x="Provider", y="Latency (sec)", color="Provider",
            color_discrete_map=PROVIDER_COLORS, points="all", template=template,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speed vs Benchmark (Chatbot Arena)")
    fig = px.scatter(
        df, x="Speed (tokens/sec)", y="Benchmark (Chatbot Arena)", color="Provider",
        size="Compute Power", hover_name="Model", color_discrete_map=PROVIDER_COLORS,
        template=template,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ================= PRICING & EFFICIENCY =================
with tab_price:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Price by Provider")
        avg_price = df.groupby("Provider", as_index=False)["Price / Million Tokens"].mean()
        fig = px.bar(
            avg_price.sort_values("Price / Million Tokens"), x="Price / Million Tokens",
            y="Provider", orientation="h", color="Provider",
            color_discrete_map=PROVIDER_COLORS, template=template,
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Price Distribution")
        fig = px.histogram(
            df, x="Price / Million Tokens", color="Open-Source", nbins=25,
            barmode="overlay", template=template,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Energy Efficiency vs Compute Power")
    fig = px.scatter(
        df, x="Compute Power", y="Energy Efficiency", color="Provider",
        size="Price / Million Tokens", hover_name="Model",
        color_discrete_map=PROVIDER_COLORS, template=template,
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Rating vs Quality Rating (bubble = count)")
    grid = (
        df.groupby(["Price Rating", "Quality Rating"]).size().reset_index(name="Count")
    )
    fig = px.scatter(
        grid, x="Price Rating", y="Quality Rating", size="Count", color="Count",
        color_continuous_scale="Blues", template=template,
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ================= COMPARE MODELS (radar + parallel coords) =================
with tab_compare:
    default_models = df.sort_values("Benchmark (MMLU)", ascending=False)["Model"].head(4).tolist()
    selected_models = st.multiselect(
        "Pick up to 6 models to compare head-to-head",
        df["Model"].tolist(), default=default_models, max_selections=6,
    )

    if selected_models:
        radar_metrics = {
            "Benchmark (MMLU)": True,
            "Benchmark (Chatbot Arena)": True,
            "Speed (tokens/sec)": True,
            "Energy Efficiency": True,
            "Latency (sec)": False,          # lower is better -> invert
            "Price / Million Tokens": False,  # lower is better -> invert
        }
        base = df_raw  # normalize against the full dataset for stable scaling
        norm = pd.DataFrame({"Model": base["Model"]})
        for metric, higher_better in radar_metrics.items():
            mn, mx = base[metric].min(), base[metric].max()
            scaled = (base[metric] - mn) / (mx - mn) if mx > mn else 0.5
            norm[metric] = scaled if higher_better else 1 - scaled

        c1, c2 = st.columns((1, 1))
        with c1:
            st.subheader("Radar — Normalized Strengths (0-1, higher = better)")
            fig = go.Figure()
            metric_names = list(radar_metrics.keys())
            for m in selected_models:
                row = norm[norm["Model"] == m].iloc[0]
                fig.add_trace(
                    go.Scatterpolar(
                        r=[row[c] for c in metric_names] + [row[metric_names[0]]],
                        theta=metric_names + [metric_names[0]],
                        fill="toself", name=m,
                    )
                )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template=template, height=470, showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Parallel Coordinates — Raw Metrics")
            pc_df = df[df["Model"].isin(selected_models)].copy()
            pc_df["ProviderCode"] = pc_df["Provider"].astype("category").cat.codes
            fig = px.parallel_coordinates(
                pc_df,
                dimensions=["Benchmark (MMLU)", "Benchmark (Chatbot Arena)",
                            "Speed (tokens/sec)", "Latency (sec)",
                            "Price / Million Tokens", "Energy Efficiency"],
                color="ProviderCode", color_continuous_scale=px.colors.qualitative.Set2,
                template=template,
            )
            fig.update_layout(height=470)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Side-by-Side Table")
        st.dataframe(
            df[df["Model"].isin(selected_models)].set_index("Model"),
            use_container_width=True,
        )
    else:
        st.info("Select at least one model above to see the comparison charts.")

# ================= CORRELATIONS =================
with tab_corr:
    st.subheader("Correlation Heatmap — Numeric Metrics")
    corr = df[NUMERIC_COLS].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", template=template,
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X metric", NUMERIC_COLS, index=NUMERIC_COLS.index("Compute Power"))
    with c2:
        y_axis = st.selectbox("Y metric", NUMERIC_COLS, index=NUMERIC_COLS.index("Benchmark (MMLU)"))
    fig = px.scatter(
        df, x=x_axis, y=y_axis, color="Provider", trendline="ols",
        hover_name="Model", color_discrete_map=PROVIDER_COLORS, template=template,
    )
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

# ================= DATA EXPLORER =================
with tab_data:
    st.subheader("Filtered Dataset")
    st.dataframe(
        df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quality Rating": st.column_config.ProgressColumn(
                "Quality Rating", min_value=1, max_value=3, format="%d"
            ),
            "Speed Rating": st.column_config.ProgressColumn(
                "Speed Rating", min_value=1, max_value=3, format="%d"
            ),
            "Price Rating": st.column_config.ProgressColumn(
                "Price Rating", min_value=1, max_value=3, format="%d"
            ),
            "Open-Source": st.column_config.CheckboxColumn("Open-Source"),
        },
    )
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_llm_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )