"""
pages/5_Outliers.py
Isolation Forest outlier detection — detect, visualise, optionally remove.
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
from outlier import detect_outliers, outlier_summary, remove_outliers
from pca import apply_pca

st.set_page_config(page_title="Outlier Detection", page_icon="⚠️", layout="wide")

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

# Dynamic display columns — use first 4 numeric cols from df_feat
num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
display_cols = num_cols[:4]

st.markdown("<div class='hero-title'>⚠️ Outlier Detection</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>Using <strong>Isolation Forest</strong> to detect anomalous customers.</p>",
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Isolation Forest Settings")
    contamination = st.slider(
        "Contamination (expected % outliers)",
        min_value=1, max_value=15, value=5, step=1, key="contamination"
    ) / 100.0
    remove_them = st.checkbox("Remove outliers from downstream analysis", value=False)

# ── Detect ────────────────────────────────────────────────────────────────────
iso_labels, iso_scores, inlier_mask = detect_outliers(X_scaled, contamination=contamination)
summary = outlier_summary(iso_labels)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(iso_labels))
c2.metric("Outliers Detected", summary["n_outliers"], delta=f"{summary['pct_outliers']} %", delta_color="inverse")
c3.metric("Inliers", summary["n_inliers"])
c4.metric("Contamination Setting", f"{contamination*100:.0f} %")

# ── PCA 2D scatter ────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Outlier Map (PCA 2D)</div>", unsafe_allow_html=True)

X_pca, _, _ = apply_pca(X_scaled, n_components=2)

# Normalize scores safely (avoid division by zero)
score_min = iso_scores.min()
score_max = iso_scores.max()
if score_max - score_min == 0:
    score_scaled = np.zeros_like(iso_scores)
else:
    score_scaled = (iso_scores - score_min) / (score_max - score_min)

df_plot = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Status": ["Outlier" if l == -1 else "Inlier" for l in iso_labels],
    "AnomalyScore": score_scaled,
})
# Add dynamic columns
for col in display_cols:
    if col in df_feat.columns:
        df_plot[col] = df_feat[col].values

fig_out = px.scatter(
    df_plot, x="PC1", y="PC2", color="Status",
    color_discrete_map={"Inlier": "#00e5cc", "Outlier": "#ff4444"},
    size="AnomalyScore",
    size_max=18,
    hover_data=display_cols + ["AnomalyScore"],
    template="plotly_dark",
    title="Isolation Forest — Inliers vs Outliers",
)

fig_out.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
    height=500,
)

st.plotly_chart(fig_out, use_container_width=True)

# ── Anomaly Score Distribution ────────────────────────────────────────────────
st.markdown("<div class='section-header'>Anomaly Score Distribution</div>", unsafe_allow_html=True)

fig_score = go.Figure()
fig_score.add_trace(go.Histogram(
    x=iso_scores[inlier_mask], nbinsx=25,
    name="Inliers", marker_color="#00e5cc", opacity=0.75,
))
fig_score.add_trace(go.Histogram(
    x=iso_scores[~inlier_mask], nbinsx=10,
    name="Outliers", marker_color="#ff4444", opacity=0.85,
))

threshold = iso_scores[~inlier_mask].max() if (~inlier_mask).any() else 0
fig_score.add_vline(
    x=threshold,
    line_dash="dash",
    line_color="#ff9900",
    annotation_text="Approx Threshold",
    annotation_font_color="#ff9900"
)

fig_score.update_layout(
    barmode="overlay",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,21,32,0.8)",
    title="Anomaly Score Histogram",
    height=350,
)

st.plotly_chart(fig_score, use_container_width=True)

# ── Feature-wise box plots ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Feature Comparison — Inliers vs Outliers</div>",
            unsafe_allow_html=True)

df_feat_copy = df_feat.copy()
df_feat_copy["Status"] = ["Outlier" if l == -1 else "Inlier" for l in iso_labels]

melt_cols = [c for c in display_cols if c in df_feat_copy.columns]
if melt_cols:
    fig_box = px.box(
        df_feat_copy.melt(id_vars="Status", value_vars=melt_cols,
                          var_name="Feature", value_name="Value"),
        x="Feature", y="Value", color="Status",
        color_discrete_map={"Inlier": "#00e5cc", "Outlier": "#ff4444"},
        template="plotly_dark",
        points="outliers",
    )
    fig_box.update_layout(height=420)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Outlier table ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Detected Outlier Records</div>", unsafe_allow_html=True)

df_outlier_display = df_feat[~inlier_mask].copy()
available_cols = [c for c in display_cols if c in df_outlier_display.columns]
df_outliers = df_outlier_display[available_cols].copy()
df_outliers["AnomalyScore"] = iso_scores[~inlier_mask].round(4)

st.dataframe(df_outliers.style.background_gradient(cmap="Reds"), use_container_width=True)

# ── Optional removal ──────────────────────────────────────────────────────────
if remove_them:
    X_clean = remove_outliers(X_scaled, inlier_mask)
    st.success(
        f"✅ Outliers removed — dataset reduced from {len(X_scaled)} → {len(X_clean)} rows. "
        f"This cleaned dataset would be used in downstream clustering."
    )
