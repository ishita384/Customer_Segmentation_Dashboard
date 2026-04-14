"""
pages/4_PCA_Visualization.py
Dimensionality reduction with PCA — visualise clusters in 2D.
Updated: session state, dynamic columns, NaN fix.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from preprocessing import full_pipeline
from clustering import run_kmeans
from pca import apply_pca, pca_summary


st.set_page_config(page_title="PCA Visualization", page_icon="📉", layout="wide")

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0e1117;color:#e0e6f0;font-family:'Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1526,#0e1117);border-right:1px solid #1e2d45}
.card{background:linear-gradient(135deg,#141c2e,#0e1520);border:1px solid #1e3050;border-radius:16px;padding:24px 28px;margin-bottom:20px;box-shadow:0 4px 24px rgba(0,180,255,.06);transition:transform .2s,box-shadow .2s}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,180,255,.14)}
.hero-title{font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#00b4ff,#00e5cc,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.section-header{font-size:1.2rem;font-weight:700;color:#00b4ff;border-left:4px solid #00e5cc;padding-left:12px;margin:24px 0 14px}
[data-testid="stMetric"]{background:linear-gradient(135deg,#141c2e,#0e1520);border:1px solid #1e3050;border-radius:12px;padding:16px 20px}
[data-testid="stMetricValue"]{color:#00e5cc;font-weight:700}
[data-testid="stMetricLabel"]{color:#7a96b8}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE FIX ────────────────────────────────────────────────────────
df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Please go to **Data Explorer** first to load a dataset.")
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipe = full_pipeline(df)
X_scaled = pipe["X_scaled"]
df_feat = pipe["featured"]
feature_cols = pipe["feature_cols"]

# NaN fix (critical)
imputer = SimpleImputer(strategy="mean")
X_scaled = imputer.fit_transform(X_scaled)

# Use actual feature_cols as component labels for PCA loadings
n_features = X_scaled.shape[1]

st.markdown("<div class='hero-title'>📉 PCA Visualization</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>PCA reduces high-dimensional data to 2D so we can <em>see</em> the clusters.</p>",
            unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    k_pca = st.slider("K-Means clusters", 2, 10, 5, key="pca_k")

    # Dynamic color options — only use columns that actually exist
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    color_options = ["Cluster"] + num_cols[:4]
    color_by = st.selectbox("Colour points by", color_options)

# ── Run PCA + Clustering ──────────────────────────────────────────────────────
X_pca, pca_obj, explained = apply_pca(X_scaled, n_components=2)
summary = pca_summary(pca_obj)
km_labels, _ = run_kmeans(X_scaled, n_clusters=k_pca)

# Build plot dataframe dynamically
df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
    "Cluster": [f"Cluster {l}" for l in km_labels],
})
# Add available numeric columns for hover
for col in num_cols[:5]:
    if col in df_feat.columns:
        df_pca[col] = df_feat[col].values

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Original Dimensions", n_features)
c2.metric("After PCA", 2)
c3.metric("PC1 Variance", f"{explained[0]:.1f}%")
c4.metric("Total Explained", f"{summary['total_explained']:.1f}%")

# ── Explained Variance Bar ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Explained Variance per Component</div>", unsafe_allow_html=True)

fig_var = go.Figure(go.Bar(
    x=[f"PC{i+1}" for i in range(len(explained))],
    y=explained,
    marker_color=["#00b4ff", "#00e5cc"],
    text=[f"{v:.1f}%" for v in explained],
    textposition="outside",
))

fig_var.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,21,32,0.8)",
    yaxis_title="Variance Explained (%)",
    title="PCA Explained Variance",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
    height=320,
)

st.plotly_chart(fig_var, use_container_width=True)

# ── Main PCA scatter ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>2D PCA Cluster Plot</div>", unsafe_allow_html=True)

hover_data_cols = [c for c in num_cols[:4] if c in df_pca.columns]

if color_by == "Cluster":
    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=hover_data_cols,
        template="plotly_dark",
        title="K-Means Clusters in PCA Space",
    )
else:
    if color_by in df_pca.columns:
        fig_pca = px.scatter(
            df_pca, x="PC1", y="PC2", color=color_by,
            color_continuous_scale="Viridis",
            hover_data=hover_data_cols + ["Cluster"],
            template="plotly_dark",
            title=f"PCA Scatter coloured by {color_by}",
        )
    else:
        st.warning(f"Column '{color_by}' not available. Falling back to Cluster coloring.")
        fig_pca = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark",
            title="K-Means Clusters in PCA Space",
        )

fig_pca.update_traces(marker_size=10, marker_opacity=0.85)

fig_pca.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    xaxis=dict(title=f"PC1 ({explained[0]:.1f}% variance)", gridcolor="#1e3050"),
    yaxis=dict(title=f"PC2 ({explained[1]:.1f}% variance)", gridcolor="#1e3050"),
    height=540,
)

st.plotly_chart(fig_pca, use_container_width=True)

# ── Feature Loadings ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Feature Loadings (How Each Feature Contributes)</div>",
            unsafe_allow_html=True)

# Use actual feature_cols for loading labels (safe — same number as PCA input columns)
loadings = pd.DataFrame(
    pca_obj.components_.T,
    index=feature_cols[:n_features],
    columns=[f"PC{i+1}" for i in range(2)]
).round(4)

fig_load = px.bar(
    loadings.reset_index(),
    x="index",
    y=["PC1", "PC2"],
    barmode="group",
    color_discrete_sequence=["#00b4ff", "#00e5cc"],
    labels={"index": "Feature", "value": "Loading"},
    template="plotly_dark",
    title="Feature Loadings on PC1 & PC2",
)

fig_load.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
    height=380,
)

st.plotly_chart(fig_load, use_container_width=True)

st.markdown("""
<div class='card'>
    <p style='color:#7a96b8'>
    📌 <strong style='color:#00e5cc'>Interpretation:</strong>
    Features with large absolute loadings have the most influence on that principal component.
    A positive loading means the feature increases along that axis; negative means it decreases.
    </p>
</div>
""", unsafe_allow_html=True)
