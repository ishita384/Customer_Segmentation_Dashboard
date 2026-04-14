"""
pages/3_Clustering.py
Interactive K-Means, DBSCAN, Random Forest, SVM, and Linear Regression clustering/modelling.
Now includes: data split, multi-model selection, dynamic columns.
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
from clustering import (
    compute_elbow, run_kmeans, run_dbscan, dbscan_summary, compute_silhouette,
    split_data, split_data_supervised,
    run_random_forest_classifier, run_svm_classifier,
    run_linear_regression, run_random_forest_regressor,
    predict_cluster,   # ✅ NEW: live single-customer prediction
)
from pca import apply_pca

st.set_page_config(page_title="Clustering", page_icon="🧠", layout="wide")

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
.stSlider [data-baseweb="slider"]{color:#00b4ff}
.stButton>button{background:linear-gradient(135deg,#0070f3,#00b4d8);color:white;border:none;border-radius:10px;padding:.5rem 1.4rem;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE FIX ────────────────────────────────────────────────────────
df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Please go to **Data Explorer** first to load a dataset.")
    st.stop()

# ── Data pipeline ─────────────────────────────────────────────────────────────
pipe = full_pipeline(df)
X_scaled = pipe["X_scaled"]
df_feat = pipe["featured"]
feature_cols = pipe["feature_cols"]

# NaN fix (critical)
imputer = SimpleImputer(strategy="mean")
X_scaled = imputer.fit_transform(X_scaled)

# Dynamic hover columns — use whatever numeric cols exist
num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
hover_cols = num_cols[:4]  # first 4 numeric columns for hover

st.markdown("<div class='hero-title'>🧠 Clustering</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>K-Means with Elbow method, DBSCAN with noise detection, and supervised models.</p>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔵 K-Means", "🟠 DBSCAN", "🤖 Supervised Models"])

# ── PCA for 2-D viz ───────────────────────────────────────────────────────────
X_pca, _, _ = apply_pca(X_scaled, n_components=2)

PALETTE = px.colors.qualitative.Bold

# ═══ K-MEANS TAB ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Elbow Method — Find Optimal K</div>", unsafe_allow_html=True)

    k_max = st.slider("Max K for Elbow", min_value=4, max_value=15, value=10, key="elbow_k")
    inertias = compute_elbow(X_scaled, k_range=range(2, k_max + 1))

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(inertias.keys()), y=list(inertias.values()),
        mode="lines+markers",
        line=dict(color="#00b4ff", width=3),
        marker=dict(size=9, color="#00e5cc", symbol="circle"),
        name="WCSS (Inertia)",
    ))
    fig_elbow.update_layout(
        title="Elbow Curve", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
        xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)",
        title_font_color="#00e5cc", font_color="#e0e6f0",
        xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
        height=380,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("""
    <div class='card'>
        <p style='color:#7a96b8;margin:0'>
        📌 <strong style='color:#00e5cc'>How to read:</strong>
        Look for the "elbow" — the point where inertia stops dropping sharply.
        Adding more clusters beyond the elbow gives diminishing returns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Run K-Means Clustering</div>", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        k_final = st.slider("Choose number of clusters (K)", min_value=2, max_value=10, value=5, key="k_final")
    with col_ctrl2:
        test_size_km = st.slider("Test Split Size (%)", min_value=10, max_value=40, value=20, key="km_split") / 100

    # DATA SPLIT
    X_train_km, X_test_km = split_data(X_scaled, test_size=test_size_km)

    km_labels, km_model = run_kmeans(X_scaled, n_clusters=k_final)
    sil_km = compute_silhouette(X_scaled, km_labels)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters (K)", k_final)
    m2.metric("Silhouette Score", f"{sil_km:.4f}")
    m3.metric("Inertia", f"{km_model.inertia_:,.1f}")
    m4.metric("Train / Test Split", f"{int((1-test_size_km)*100)}% / {int(test_size_km*100)}%")

    # Cluster scatter (PCA 2D)
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": [f"Cluster {l}" for l in km_labels],
    })
    # Add dynamic hover columns
    for hc in hover_cols:
        if hc in df_feat.columns:
            df_plot[hc] = df_feat[hc].values

    fig_km = px.scatter(
        df_plot, x="PC1", y="PC2", color="Cluster",
        hover_data=hover_cols,
        color_discrete_sequence=PALETTE,
        template="plotly_dark",
        title=f"K-Means Clusters (K={k_final}) — PCA 2D View",
    )
    fig_km.update_traces(marker_size=8, marker_opacity=0.85)
    fig_km.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
        title_font_color="#00e5cc", font_color="#e0e6f0",
        xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
        height=500,
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # Cluster profile table — dynamic columns
    st.markdown("<div class='section-header'>Cluster Profiles</div>", unsafe_allow_html=True)
    df_feat_copy = df_feat.copy()
    df_feat_copy["Cluster"] = km_labels
    profile_cols = [c for c in num_cols if c in df_feat_copy.columns][:6]
    profile = df_feat_copy.groupby("Cluster")[profile_cols].mean().round(2)
    st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)

# ═══ DBSCAN TAB ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>DBSCAN Parameters</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        eps_val = st.slider("eps (neighbourhood radius)", 0.1, 3.0, 0.8, 0.05, key="eps")
    with col_b:
        min_s = st.slider("min_samples (core point threshold)", 2, 20, 5, 1, key="min_s")

    db_labels = run_dbscan(X_scaled, eps=eps_val, min_samples=min_s)
    summary = dbscan_summary(db_labels)
    sil_db = compute_silhouette(X_scaled, db_labels)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters Found", summary["n_clusters"])
    m2.metric("Noise Points", summary["n_noise"])
    m3.metric("Silhouette Score", f"{sil_db:.4f}" if not np.isnan(sil_db) else "N/A")
    m4.metric("Total Points", len(db_labels))

    label_str = ["Noise" if l == -1 else f"Cluster {l}" for l in db_labels]

    df_db = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Label": label_str})
    for hc in hover_cols:
        if hc in df_feat.columns:
            df_db[hc] = df_feat[hc].values

    color_map = {f"Cluster {i}": PALETTE[i % len(PALETTE)] for i in range(summary["n_clusters"])}
    color_map["Noise"] = "#ff4444"

    fig_db = px.scatter(
        df_db, x="PC1", y="PC2", color="Label",
        hover_data=hover_cols,
        color_discrete_map=color_map,
        template="plotly_dark",
        title=f"DBSCAN (eps={eps_val}, min_samples={min_s}) — PCA 2D View",
    )
    fig_db.update_traces(marker_size=8, marker_opacity=0.85)
    fig_db.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
        title_font_color="#00e5cc", font_color="#e0e6f0",
        xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
        height=500,
    )
    st.plotly_chart(fig_db, use_container_width=True)

    st.markdown("""
    <div class='card'>
        <p style='color:#7a96b8'>
        📌 <strong style='color:#ff4444'>Red points</strong> are <strong>noise</strong> — 
        DBSCAN could not assign them to any cluster.<br>
        Increase <strong>eps</strong> or decrease <strong>min_samples</strong> to reduce noise. <br>
        DBSCAN discovers clusters of arbitrary shape — unlike K-Means which assumes spherical clusters.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══ SUPERVISED MODELS TAB (NEW) ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Model Selection & Training</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <p style='color:#7a96b8'>
    Use cluster labels as targets for supervised learning — predict which cluster a customer belongs to,
    or predict a numeric feature using regression. Choose your model and target below.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest (Classifier)", "SVM Classifier", "Linear Regression", "Random Forest (Regressor)"],
            key="model_select"
        )
    with ctrl2:
        k_for_labels = st.slider("K-Means clusters (for target labels)", 2, 8, 5, key="k_sup")
    with ctrl3:
        test_size_sup = st.slider("Test Split (%)", 10, 40, 20, key="sup_split") / 100

    # Generate cluster labels as classification target
    km_labels_sup, _ = run_kmeans(X_scaled, n_clusters=k_for_labels)
    y_class = km_labels_sup

    # For regression — let user pick target column
    is_regression = "Regressor" in model_type or "Regression" in model_type

    if is_regression:
        reg_target_col = st.selectbox("Regression target column", num_cols, key="reg_target")
        y_reg = df_feat[reg_target_col].fillna(df_feat[reg_target_col].median()).values
    else:
        reg_target_col = None
        y_reg = None

    # ── Train button ──────────────────────────────────────────────────────────
    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training model..."):

            if is_regression:
                X_train, X_test, y_train, y_test = split_data_supervised(
                    X_scaled, y_reg, test_size=test_size_sup
                )
                if "Linear Regression" in model_type:
                    model, r2, mse = run_linear_regression(X_train, X_test, y_train, y_test)
                    model_name = "Linear Regression"
                else:
                    model, r2, mse = run_random_forest_regressor(X_train, X_test, y_train, y_test)
                    model_name = "Random Forest Regressor"

                st.success(f"✅ {model_name} trained successfully!")
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("MSE", f"{mse:.4f}")
                m3.metric("Train Samples", len(X_train))

                st.markdown(f"""
                <div class='card'>
                    <p style='color:#7a96b8'>
                    Model: <strong style='color:#00e5cc'>{model_name}</strong><br>
                    Target: <strong style='color:#00e5cc'>{reg_target_col}</strong><br>
                    R² = {r2:.4f} &nbsp;|&nbsp; MSE = {mse:.4f}<br>
                    Train size = {len(X_train)} &nbsp;|&nbsp; Test size = {len(X_test)}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Save regressor to session state (no cluster prediction form for regression)
                st.session_state["trained_model"] = None
                st.session_state["trained_model_name"] = model_name
                st.session_state["trained_model_type"] = "regression"

            else:
                X_train, X_test, y_train, y_test = split_data_supervised(
                    X_scaled, y_class, test_size=test_size_sup
                )
                if "Random Forest" in model_type:
                    model, acc, report = run_random_forest_classifier(X_train, X_test, y_train, y_test)
                    model_name = "Random Forest Classifier"
                else:
                    model, acc = run_svm_classifier(X_train, X_test, y_train, y_test)
                    report = None
                    model_name = "SVM Classifier"

                st.success(f"✅ {model_name} trained successfully!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("Train Samples", len(X_train))
                m3.metric("Test Samples", len(X_test))

                st.markdown(f"""
                <div class='card'>
                    <p style='color:#7a96b8'>
                    Model: <strong style='color:#00e5cc'>{model_name}</strong><br>
                    Target: Cluster labels (K={k_for_labels})<br>
                    Accuracy = {acc:.4f}<br>
                    Train size = {len(X_train)} &nbsp;|&nbsp; Test size = {len(X_test)}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Feature importances (Random Forest only)
                if model_name == "Random Forest Classifier":
                    st.markdown("<div class='section-header'>Feature Importances</div>", unsafe_allow_html=True)
                    feat_imp = pd.DataFrame({
                        "Feature": feature_cols,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False)

                    fig_imp = px.bar(
                        feat_imp, x="Feature", y="Importance",
                        color="Importance", color_continuous_scale="Blues",
                        template="plotly_dark",
                        title="Random Forest Feature Importances",
                    )
                    fig_imp.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
                        title_font_color="#00e5cc", font_color="#e0e6f0", height=350,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # ✅ Save classifier + metadata to session_state for prediction form below
                st.session_state["trained_model"]       = model
                st.session_state["trained_model_name"]  = model_name
                st.session_state["trained_model_type"]  = "classifier"
                st.session_state["trained_k"]           = k_for_labels
                st.session_state["trained_scaler"]      = pipe["scaler"]
                st.session_state["trained_feature_cols"] = feature_cols
                st.session_state["trained_df_feat"]     = df_feat

    else:
        st.info("👆 Configure settings above and click **Train Model** to run.")

    # ══════════════════════════════════════════════════════════════════════════
    # ── LIVE PREDICTION FORM — appears only after a classifier is trained ────
    # ══════════════════════════════════════════════════════════════════════════
    if (
        st.session_state.get("trained_model") is not None
        and st.session_state.get("trained_model_type") == "classifier"
    ):
        st.markdown("---")
        st.markdown("<div class='section-header'>🔮 Predict Cluster for a New Customer</div>",
                    unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <p style='color:#7a96b8;margin:0'>
            Enter customer details below. The trained
            <strong style='color:#00e5cc'>{}</strong> will predict
            which of the <strong style='color:#00e5cc'>{} clusters</strong>
            this customer belongs to.
            </p>
        </div>
        """.format(
            st.session_state["trained_model_name"],
            st.session_state["trained_k"]
        ), unsafe_allow_html=True)

        # Retrieve saved metadata
        saved_feature_cols = st.session_state["trained_feature_cols"]
        saved_df_feat      = st.session_state["trained_df_feat"]
        saved_scaler       = st.session_state["trained_scaler"]
        saved_model        = st.session_state["trained_model"]

        # Build input widgets — one per feature, laid out 3 per row
        input_values = {}
        cols_per_row = 3
        feat_chunks  = [
            saved_feature_cols[i : i + cols_per_row]
            for i in range(0, len(saved_feature_cols), cols_per_row)
        ]

        for chunk in feat_chunks:
            row_cols = st.columns(cols_per_row)
            for col_widget, feat in zip(row_cols, chunk):
                with col_widget:
                    if feat in saved_df_feat.columns:
                        col_data   = saved_df_feat[feat].dropna()
                        col_min    = float(col_data.min())
                        col_max    = float(col_data.max())
                        col_mean   = float(col_data.mean())
                        col_median = float(col_data.median())

                        # Use integer step for binary / label-encoded cols,
                        # float step for continuous ones
                        is_binary = sorted(col_data.unique().tolist()) in ([0, 1], [0.0, 1.0])
                        step      = 1.0 if (col_max - col_min) <= 5 else round((col_max - col_min) / 100, 2)

                        input_values[feat] = st.number_input(
                            label=f"{feat}",
                            min_value=col_min,
                            max_value=col_max,
                            value=col_median,
                            step=step,
                            key=f"pred_input_{feat}",
                            help=f"Range: {col_min:.1f} – {col_max:.1f}  |  Dataset mean: {col_mean:.1f}",
                        )
                    else:
                        # Feature was engineered and not in original df (e.g. Gender_Encoded)
                        input_values[feat] = st.number_input(
                            label=f"{feat}",
                            value=0.0,
                            step=1.0,
                            key=f"pred_input_{feat}",
                            help="Encoded feature — enter 0 or 1",
                        )

        # ── Predict button ────────────────────────────────────────────────────
        pred_col1, pred_col2 = st.columns([1, 3])
        with pred_col1:
            predict_clicked = st.button("🎯 Predict Cluster", key="predict_btn")

        if predict_clicked:
            try:
                predicted_cluster, proba = predict_cluster(
                    model=saved_model,
                    scaler=saved_scaler,
                    input_dict=input_values,
                    feature_cols=saved_feature_cols,
                )

                # ── Result banner ─────────────────────────────────────────────
                st.markdown(f"""
                <div class='card' style='border-color:#00e5cc; border-width:2px; text-align:center;'>
                    <p style='font-size:1rem;color:#7a96b8;margin:0'>Predicted Segment</p>
                    <p style='font-size:3rem;font-weight:900;
                              background:linear-gradient(90deg,#00b4ff,#00e5cc);
                              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                              margin:4px 0;'>
                        Cluster {predicted_cluster}
                    </p>
                    <p style='color:#7a96b8;margin:0;font-size:0.9rem'>
                        Model: {st.session_state["trained_model_name"]} &nbsp;|&nbsp;
                        K = {st.session_state["trained_k"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability bar chart (if available) ──────────────────────
                if proba:
                    st.markdown("<div class='section-header'>Cluster Probabilities</div>",
                                unsafe_allow_html=True)

                    prob_df = pd.DataFrame({
                        "Cluster": [f"Cluster {k}" for k in proba.keys()],
                        "Probability": list(proba.values()),
                    }).sort_values("Probability", ascending=False)

                    # Highlight predicted cluster
                    colors = [
                        "#00e5cc" if c == f"Cluster {predicted_cluster}" else "#1e4070"
                        for c in prob_df["Cluster"]
                    ]

                    fig_prob = go.Figure(go.Bar(
                        x=prob_df["Cluster"],
                        y=prob_df["Probability"],
                        marker_color=colors,
                        text=[f"{v:.1%}" for v in prob_df["Probability"]],
                        textposition="outside",
                    ))
                    fig_prob.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(14,21,32,0.8)",
                        yaxis=dict(range=[0, 1.1], title="Probability",
                                   gridcolor="#1e3050"),
                        xaxis=dict(gridcolor="#1e3050"),
                        title="How confident is the model?",
                        title_font_color="#00e5cc",
                        font_color="#e0e6f0",
                        height=360,
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                # ── Input summary ──────────────────────────────────────────────
                st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
                summary_df = pd.DataFrame(
                    list(input_values.items()),
                    columns=["Feature", "Your Input"]
                )
                summary_df["Your Input"] = summary_df["Your Input"].round(4)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
