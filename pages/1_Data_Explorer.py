"""
pages/1_Data_Explorer.py
Dynamic dataset explorer (ANY dataset supported)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from preprocessing import load_data

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# ── UI STYLE (FROM YOUR REQUESTED DESIGN) ────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0e1117;color:#e0e6f0;font-family:'Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1526,#0e1117);border-right:1px solid #1e2d45}
.card{background:linear-gradient(135deg,#141c2e,#0e1520);border:1px solid #1e3050;border-radius:16px;padding:24px 28px;margin-bottom:20px;box-shadow:0 4px 24px rgba(0,180,255,.06);transition:transform .2s,box-shadow .2s}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,180,255,.14)}
.hero-title{font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#00b4ff,#00e5cc,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.section-header{font-size:1.2rem;font-weight:700;color:#00b4ff;border-left:4px solid #00e5cc;padding-left:12px;margin:24px 0 14px}
[data-testid="stMetric"]{background:linear-gradient(135deg,#141c2e,#0e1520);border:1px solid #1e3050;border-radius:12px;padding:16px 20px}
[data-testid="stMetricValue"]{color:#00e5cc;font-weight:700}
[data-testid="stMetricLabel"]{color:#7a96b8}
</style>
""", unsafe_allow_html=True)

# ── DATA SOURCE (UNCHANGED FUNCTIONALITY) ───────────────────────────────────
option = st.radio(
    "Choose Dataset Source:",
    ["Use Default Dataset", "Upload Your Own"],
    horizontal=True
)

uploaded_file = None

if option == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ── LOAD DATA (SAFE + SAME LOGIC) ────────────────────────────────────────────
if option == "Use Default Dataset":
    df = load_data()
    st.session_state["df"] = df

else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
    else:
        st.warning("Please upload dataset")
        st.stop()

# ── AUTO DETECTION ───────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>📊 Data Explorer</div>", unsafe_allow_html=True)

# ── KPI (SAFE) ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Columns", df.shape[1])

if num_cols:
    selected_metric = st.selectbox("Select numeric column", num_cols)
    c3.metric("Average", f"{df[selected_metric].mean():.2f}")
    c4.metric("Max", f"{df[selected_metric].max():.2f}")
else:
    c3.metric("Average", "N/A")
    c4.metric("Max", "N/A")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Raw Data",
    "📊 Distributions",
    "🔥 Correlation",
    "🔍 Pairplot"
])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Raw Dataset</div>", unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)

    st.write("Shape:", df.shape)
    st.write("Data Types:")
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Statistics</div>", unsafe_allow_html=True)

    st.dataframe(df.describe(include='all'), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Distributions</div>", unsafe_allow_html=True)

    if num_cols:
        selected_features = st.multiselect(
            "Select features",
            num_cols,
            default=num_cols[:min(3, len(num_cols))]
        )

        if selected_features:
            cols = st.columns(len(selected_features))
            for i, col in enumerate(selected_features):
                fig = px.histogram(df, x=col, template="plotly_dark")
                cols[i].plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns available")

    if cat_cols:
        selected_cat = st.selectbox("Select categorical column", cat_cols)

        counts = df[selected_cat].value_counts().reset_index()
        counts.columns = [selected_cat, "Count"]

        fig = px.pie(counts, names=selected_cat, values="Count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Correlation</div>", unsafe_allow_html=True)

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns")

# ── TAB 4 ────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Pairplot</div>", unsafe_allow_html=True)

    if len(num_cols) >= 2:
        color_col = None
        if cat_cols:
            color_col = st.selectbox("Color by", [None] + cat_cols)

        fig = px.scatter_matrix(
            df,
            dimensions=num_cols[:min(4, len(num_cols))],
            color=color_col,
            template="plotly_dark"
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns")
