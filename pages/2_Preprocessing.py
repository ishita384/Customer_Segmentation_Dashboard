"""
pages/2_Preprocessing.py
Show each step of the data preprocessing pipeline interactively.
Now includes: Variance Threshold and Correlation-based Feature Selection.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from preprocessing import (
    introduce_missing, handle_missing,
    encode_categoricals, engineer_features, scale_features, get_feature_cols,
    apply_variance_threshold, apply_correlation_filter,
)

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

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
.step-badge{display:inline-block;background:linear-gradient(90deg,#0070f3,#00b4d8);color:white;border-radius:20px;padding:3px 14px;font-size:.8rem;font-weight:700;margin-bottom:10px}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-title'>⚙️ Data Preprocessing</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>Step-by-step data cleaning, encoding, feature engineering, scaling, and feature selection.</p>",
            unsafe_allow_html=True)

# ── SESSION STATE FIX ────────────────────────────────────────────────────────
df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Please go to **Data Explorer** first to load a dataset.")
    st.stop()

df_raw = df

# ── STEP 1 — Missing values ──────────────────────────────────────────────────
st.markdown("<div class='section-header'>Step 1 — Handle Missing Values</div>", unsafe_allow_html=True)
df_missing = introduce_missing(df_raw)
missing_counts = df_missing.isnull().sum()

col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-badge'>Before Cleaning</span>", unsafe_allow_html=True)
    st.dataframe(df_missing.head(20), use_container_width=True, height=300)
    st.markdown(f"<p style='color:#ff6b6b'>❌ Missing values: {missing_counts.sum()}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    df_clean = handle_missing(df_missing)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-badge'>After Cleaning</span>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(20), use_container_width=True, height=300)
    st.markdown(f"<p style='color:#00e5cc'>✅ Missing values after fill: {df_clean.isnull().sum().sum()}</p>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

fig_missing = px.bar(
    x=missing_counts.index, y=missing_counts.values,
    labels={"x": "Column", "y": "Missing Count"},
    color=missing_counts.values,
    color_continuous_scale="Reds",
    template="plotly_dark",
    title="Missing Values per Column",
)
fig_missing.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6f0",
                          title_font_color="#00e5cc")
st.plotly_chart(fig_missing, use_container_width=True)

# ── STEP 2 — Encoding ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Step 2 — Encode Categorical Variables</div>", unsafe_allow_html=True)
df_enc = encode_categoricals(df_clean)

cat_cols_original = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
new_encoded_cols = [c for c in df_enc.columns if c.endswith("_Encoded")]

if cat_cols_original:
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>Original Categorical Columns</span>", unsafe_allow_html=True)
        id_col = [c for c in df_clean.columns if "id" in c.lower() or "ID" in c]
        show_cols = (id_col[:1] + cat_cols_original) if id_col else cat_cols_original
        st.dataframe(df_clean[show_cols].head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>After Label Encoding</span>", unsafe_allow_html=True)
        id_col2 = [c for c in df_enc.columns if "id" in c.lower() or "ID" in c]
        show_enc = (id_col2[:1] + new_encoded_cols) if id_col2 else new_encoded_cols
        st.dataframe(df_enc[show_enc].head(10), use_container_width=True)
        st.markdown("<p style='color:#7a96b8;font-size:.85rem'>Each category → integer (Label Encoding)</p>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No categorical columns found — skipping encoding step.")

# ── STEP 3 — Feature Engineering ────────────────────────────────────────────
st.markdown("<div class='section-header'>Step 3 — Feature Engineering</div>", unsafe_allow_html=True)
df_feat = engineer_features(df_enc)

new_cols = [c for c in df_feat.columns if c not in df_enc.columns]
st.markdown("<div class='card'>", unsafe_allow_html=True)
if new_cols:
    desc_parts = []
    if "TotalValue" in new_cols:
        desc_parts.append("• <strong style='color:#00e5cc'>TotalValue</strong> = AnnualIncome × SpendingScore / 100 — proxy for customer lifetime value")
    if "AgeGroup_Encoded" in new_cols:
        desc_parts.append("• <strong style='color:#00e5cc'>AgeGroup_Encoded</strong> — Age binned: 0=Young (≤30), 1=Middle (31–50), 2=Senior (51+)")
    if desc_parts:
        st.markdown(f"<p style='color:#7a96b8'>New features created:<br>{'<br>'.join(desc_parts)}</p>",
                    unsafe_allow_html=True)
    all_display = [c for c in df_feat.columns if c not in
                   df_clean.select_dtypes(include="object").columns.tolist()]
    st.dataframe(df_feat[all_display[:8]].head(15), use_container_width=True)
else:
    st.info("No new features were engineered for this dataset (requires AnnualIncome, SpendingScore, or Age columns).")
    st.dataframe(df_feat.head(15), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── STEP 4 — Scaling ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Step 4 — Feature Scaling (StandardScaler)</div>", unsafe_allow_html=True)

# Dynamic feature columns — no hardcoded names
feature_cols = get_feature_cols(df_feat)

# Impute before scaling (NaN fix)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_feat[feature_cols])
df_feat_clean = df_feat.copy()
df_feat_clean[feature_cols] = X_imputed

X_scaled, scaler = scale_features(df_feat_clean, feature_cols)
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

col5, col6 = st.columns(2)
with col5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-badge'>Before Scaling</span>", unsafe_allow_html=True)
    st.dataframe(df_feat_clean[feature_cols].describe().T.round(2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col6:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-badge'>After StandardScaler (mean≈0, std≈1)</span>", unsafe_allow_html=True)
    st.dataframe(df_scaled.describe().T.round(2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Box plot comparison — dynamic columns, show up to first 4
display_feat = feature_cols[:4]
fig_box = go.Figure()
for col in display_feat:
    fig_box.add_trace(go.Box(y=df_feat_clean[col], name=f"{col} (raw)", marker_color="#0070f3"))
    fig_box.add_trace(go.Box(y=df_scaled[col], name=f"{col} (scaled)", marker_color="#00e5cc"))
fig_box.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    title="Feature Distributions: Raw vs Scaled",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    height=420,
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header' style='margin-top:0'>Final Feature List</div>", unsafe_allow_html=True)
for f in feature_cols:
    st.markdown(f"• <code style='color:#00e5cc'>{f}</code>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── STEP 5 — Feature Selection (NEW) ─────────────────────────────────────────
st.markdown("<div class='section-header'>Step 5 — Feature Selection</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#7a96b8'>
Feature selection reduces dimensionality by removing low-value features:<br>
• <strong style='color:#00e5cc'>Variance Threshold</strong> — removes features with near-zero variance (carry no information)<br>
• <strong style='color:#00e5cc'>Correlation Filter</strong> — removes redundant features that are highly correlated with each other
</p>
""", unsafe_allow_html=True)

fs_col1, fs_col2 = st.columns(2)
with fs_col1:
    var_threshold = st.slider("Variance Threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                               help="Features with variance below this value are removed")
with fs_col2:
    corr_threshold = st.slider("Correlation Threshold", min_value=0.5, max_value=1.0, value=0.95, step=0.01,
                                help="Features correlated above this value are deduplicated")

# Apply variance threshold
after_var = apply_variance_threshold(df_feat_clean, feature_cols, threshold=var_threshold)
# Apply correlation filter
after_corr = apply_correlation_filter(df_feat_clean, after_var, corr_threshold=corr_threshold)

m1, m2, m3 = st.columns(3)
m1.metric("Original Features", len(feature_cols))
m2.metric("After Variance Filter", len(after_var))
m3.metric("After Correlation Filter", len(after_corr))

st.markdown("<p style='color:#7a96b8; margin-top:12px'><strong style='color:#00e5cc'>Selected Features:</strong></p>",
            unsafe_allow_html=True)
for f in after_corr:
    st.markdown(f"• <code style='color:#00e5cc'>{f}</code>", unsafe_allow_html=True)

removed = [f for f in feature_cols if f not in after_corr]
if removed:
    st.markdown("<p style='color:#ff6b6b; margin-top:8px'><strong>Removed Features:</strong></p>",
                unsafe_allow_html=True)
    for f in removed:
        st.markdown(f"• <code style='color:#ff6b6b'>{f}</code>", unsafe_allow_html=True)

# Save selected features to session state for use downstream
st.session_state["selected_feature_cols"] = after_corr

st.markdown("</div>", unsafe_allow_html=True)
