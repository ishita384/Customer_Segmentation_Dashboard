"""
pages/7_Insights.py
Business insights derived from cluster profiles — simple rule-based logic.
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

st.set_page_config(page_title="Insights", page_icon="💡", layout="wide")

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
.insight-card{background:linear-gradient(135deg,#141c2e,#0e1520);border:1px solid #1e3050;border-radius:16px;padding:20px 24px;margin-bottom:16px;border-left:4px solid}
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

# NaN fix
imputer = SimpleImputer(strategy="mean")
X_scaled = imputer.fit_transform(X_scaled)

st.markdown("<div class='hero-title'>💡 Business Insights</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>Rule-based business recommendations derived from cluster profiles.</p>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Controls")
    k_ins = st.slider("Number of Clusters", 2, 10, 5, key="k_ins")

km_labels, _ = run_kmeans(X_scaled, n_clusters=k_ins)

df_feat_copy = df_feat.copy()
df_feat_copy["Cluster"] = km_labels

# ── Dynamic aggregation — use whatever numeric columns exist ──────────────────
num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()

# Build agg dict dynamically using first numeric columns available
agg_cols = num_cols[:5]  # up to 5 numeric columns

agg_dict = {"Cluster": ("Cluster", "first")}  # placeholder
agg_dict = {col: (col, "mean") for col in agg_cols}
agg_dict["Count"] = (agg_cols[0], "count")

profile = df_feat_copy.groupby("Cluster").agg(
    **agg_dict
).round(2).reset_index()

# For segmentation logic: try to use AnnualIncome/SpendingScore if available
# Otherwise fall back to first two numeric columns
income_col = "AnnualIncome" if "AnnualIncome" in profile.columns else (agg_cols[0] if agg_cols else None)
spending_col = "SpendingScore" if "SpendingScore" in profile.columns else (agg_cols[1] if len(agg_cols) > 1 else income_col)

INCOME_MED = profile[income_col].median() if income_col else 0
SPEND_MED  = profile[spending_col].median() if spending_col else 0


def label_segment(row):
    high_income   = row[income_col] >= INCOME_MED if income_col else True
    high_spending = row[spending_col] >= SPEND_MED if spending_col else True
    if high_income and high_spending:
        return "💎 Premium / VIP"
    elif high_income and not high_spending:
        return "💼 Careful Spender"
    elif not high_income and high_spending:
        return "🛍️ Impulsive Buyer"
    else:
        return "💸 Budget Conscious"


def get_strategy(segment: str) -> str:
    return {
        "💎 Premium / VIP":
            "Offer exclusive loyalty rewards, premium memberships, and early access to new products.",
        "💼 Careful Spender":
            "Target with value-for-money messaging, quality guarantees, and investment-style products.",
        "🛍️ Impulsive Buyer":
            "Flash sales, limited-time offers, and trendy products with strong visual marketing.",
        "💸 Budget Conscious":
            "Discount campaigns, bundled deals, and referral programmes to increase engagement.",
    }.get(segment, "No strategy available.")


def get_color(segment: str) -> str:
    return {
        "💎 Premium / VIP": "#7c3aed",
        "💼 Careful Spender": "#0070f3",
        "🛍️ Impulsive Buyer": "#00b4d8",
        "💸 Budget Conscious": "#00e5cc",
    }.get(segment, "#1e3050")


profile["Segment"] = profile.apply(label_segment, axis=1)
profile["Strategy"] = profile["Segment"].apply(get_strategy)

# ── KPI row ─────────────────────────────────────────────────────────────────
cols = st.columns(k_ins)
for i, row in profile.iterrows():
    with cols[i % k_ins]:
        segment_short = row["Segment"].split(" ", 1)[-1] if " " in row["Segment"] else row["Segment"]
        st.metric(f"Cluster {row['Cluster']}", segment_short, f"{row['Count']} customers")

# ── Segment Detail Cards ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Cluster Segment Details</div>", unsafe_allow_html=True)

for _, row in profile.iterrows():
    color = get_color(row["Segment"])
    metric_lines = ""
    for col in agg_cols:
        if col in row:
            metric_lines += f"• <strong style='color:#00e5cc'>{col}:</strong> {row[col]:.1f}<br>"

    st.markdown(f"""
    <div class='insight-card' style='border-left-color:{color}'>
        <div style='display:flex;justify-content:space-between;align-items:center'>
            <span style='font-size:1.1rem;font-weight:700;color:{color}'>
                Cluster {int(row['Cluster'])} — {row['Segment']}
            </span>
            <span style='color:#7a96b8;font-size:.85rem'>{int(row['Count'])} customers</span>
        </div>
        <div style='margin:10px 0;color:#7a96b8;font-size:.9rem'>
            {metric_lines}
        </div>
        <div style='background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 14px;margin-top:8px'>
            <span style='color:#e0e6f0;font-size:.9rem'>
                📌 <strong>Strategy:</strong> {row['Strategy']}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Cluster Profile Table ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Full Cluster Profile Summary</div>", unsafe_allow_html=True)
display_profile = profile.drop(columns=["Strategy"], errors="ignore")
st.dataframe(display_profile.style.background_gradient(cmap="Blues", subset=agg_cols[:3]),
             use_container_width=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Cluster Radar Chart</div>", unsafe_allow_html=True)

radar_cols = agg_cols[:4]  # use first 4 numeric cols for radar
if len(radar_cols) >= 3:
    fig_radar = go.Figure()
    PALETTE = px.colors.qualitative.Bold

    for i, row in profile.iterrows():
        values = [row[c] for c in radar_cols]
        # Normalize 0–1 for radar shape
        col_maxes = [profile[c].max() for c in radar_cols]
        values_norm = [v / m if m > 0 else 0 for v, m in zip(values, col_maxes)]
        values_norm.append(values_norm[0])  # close the polygon
        cats = radar_cols + [radar_cols[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=values_norm,
            theta=cats,
            fill="toself",
            name=f"Cluster {int(row['Cluster'])} — {row['Segment'].split(' ',1)[-1]}",
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e3050"),
            angularaxis=dict(gridcolor="#1e3050"),
            bgcolor="rgba(14,21,32,0.8)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#e0e6f0")),
        title="Cluster Profiles — Normalized Radar",
        title_font_color="#00e5cc",
        font_color="#e0e6f0",
        height=500,
    )

    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Need at least 3 numeric features for radar chart.")

# ── Bar Chart: Cluster Size ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>Cluster Sizes</div>", unsafe_allow_html=True)

fig_size = px.bar(
    profile,
    x="Cluster",
    y="Count",
    color="Segment",
    color_discrete_sequence=px.colors.qualitative.Bold,
    text="Count",
    template="plotly_dark",
    title="Number of Customers per Cluster",
)
fig_size.update_traces(textposition="outside")
fig_size.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
    title_font_color="#00e5cc", font_color="#e0e6f0",
    xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
    height=350,
)
st.plotly_chart(fig_size, use_container_width=True)
