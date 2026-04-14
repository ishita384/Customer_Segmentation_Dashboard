"""
pages/6_Evaluation.py
Silhouette Score evaluation, K-Fold cross-validation, and hyperparameter tuning.
Updated: session state, K-Fold CV, GridSearchCV, dynamic columns.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocessing import full_pipeline, scale_features
from clustering import run_kmeans, compute_silhouette, compute_elbow
from evaluation import (
    compare_feature_sets, silhouette_per_sample, interpret_silhouette,
    kfold_cross_validate, tune_kmeans, tune_random_forest, tune_svm,
)

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")

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

st.markdown("<div class='hero-title'>📈 Evaluation</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a96b8'>Silhouette Score analysis, K-Fold cross-validation, and hyperparameter tuning.</p>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Evaluation Settings")
    k_eval = st.slider("K for evaluation", 2, 10, 5, key="k_eval")
    selected_feats = st.multiselect(
        "Selected Features (subset)",
        options=feature_cols,
        default=feature_cols[:2] if len(feature_cols) >= 2 else feature_cols,
    )

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Silhouette Analysis",
    "🔁 K-Fold Validation",
    "⚙️ Hyperparameter Tuning"
])

# ═══ TAB 1: SILHOUETTE ════════════════════════════════════════════════════════
with tab1:
    # ── Silhouette for all K ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Silhouette Score vs K</div>", unsafe_allow_html=True)

    sil_scores = {}
    for k in range(2, 11):
        labs, _ = run_kmeans(X_scaled, n_clusters=k)
        sil_scores[k] = compute_silhouette(X_scaled, labs)

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=list(sil_scores.keys()), y=list(sil_scores.values()),
        mode="lines+markers",
        line=dict(color="#00e5cc", width=3),
        marker=dict(size=9, color="#00b4ff"),
    ))

    best_k = max(sil_scores, key=lambda k: sil_scores[k] if not np.isnan(sil_scores[k]) else -1)

    fig_sil.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="#7c3aed",
        annotation_text=f"Best K={best_k}",
        annotation_font_color="#7c3aed"
    )

    fig_sil.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,21,32,0.8)",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        height=380,
        title_font_color="#00e5cc",
        font_color="#e0e6f0",
        xaxis=dict(gridcolor="#1e3050"),
        yaxis=dict(gridcolor="#1e3050"),
    )

    st.plotly_chart(fig_sil, use_container_width=True)

    # ── Per-sample silhouette ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Per-Sample Silhouette Plot</div>", unsafe_allow_html=True)

    km_labels, _ = run_kmeans(X_scaled, n_clusters=k_eval)
    sample_scores = silhouette_per_sample(X_scaled, km_labels)

    df_sil = pd.DataFrame({
        "CustomerIndex": np.arange(len(km_labels)),
        "SilhouetteScore": sample_scores,
        "Cluster": [f"Cluster {l}" for l in km_labels],
    }).sort_values(["Cluster", "SilhouetteScore"])

    fig_bar = px.bar(
        df_sil,
        x="CustomerIndex",
        y="SilhouetteScore",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
    )

    fig_bar.add_hline(y=0, line_color="#ff4444", line_dash="dash")
    fig_bar.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(14,21,32,0.8)",
                          font_color="#e0e6f0",
                          xaxis=dict(gridcolor="#1e3050"),
                          yaxis=dict(gridcolor="#1e3050"))

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class='card'>
    <p style='color:#7a96b8;margin:0'>
    📌 Bars above 0 = well-classified points. Below 0 = possible misclassified points.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature set comparison ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Feature Set Comparison: All vs Selected</div>",
                unsafe_allow_html=True)

    if len(selected_feats) >= 1:
        valid_feats = [f for f in selected_feats if f in df_feat.columns]
        if len(valid_feats) == 0:
            st.warning("No valid selected features found. Using all features instead.")
            valid_feats = feature_cols

        X_selected, _ = scale_features(df_feat, valid_feats)
        X_selected = imputer.fit_transform(X_selected)

        if X_selected.shape[1] == 0:
            X_selected = X_scaled

        comparison = compare_feature_sets(X_scaled, X_selected, n_clusters=k_eval)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#00b4ff'>All Features ({len(feature_cols)})</h4>", unsafe_allow_html=True)
            all_res = comparison["All Features"]
            st.metric("Silhouette Score", all_res["silhouette"])
            st.metric("Inertia", f"{all_res['inertia']:,}")
            st.markdown(f"<p style='color:#7a96b8'>{interpret_silhouette(all_res['silhouette'])}</p>",
                        unsafe_allow_html=True)
            for f in feature_cols:
                st.markdown(f"• <code style='color:#00e5cc'>{f}</code>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#00e5cc'>Selected Features ({len(valid_feats)})</h4>",
                        unsafe_allow_html=True)
            sel_res = comparison["Selected Features"]
            st.metric("Silhouette Score", sel_res["silhouette"])
            st.metric("Inertia", f"{sel_res['inertia']:,}")
            st.markdown(f"<p style='color:#7a96b8'>{interpret_silhouette(sel_res['silhouette'])}</p>",
                        unsafe_allow_html=True)
            for f in valid_feats:
                st.markdown(f"• <code style='color:#00e5cc'>{f}</code>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        fig_comp = go.Figure(go.Bar(
            x=["All Features", "Selected Features"],
            y=[all_res["silhouette"], sel_res["silhouette"]],
            marker_color=["#00b4ff", "#00e5cc"],
            text=[f"{all_res['silhouette']:.4f}", f"{sel_res['silhouette']:.4f}"],
            textposition="outside",
        ))

        fig_comp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,21,32,0.8)",
            yaxis_title="Silhouette Score",
            font_color="#e0e6f0",
            xaxis=dict(gridcolor="#1e3050"),
            yaxis=dict(gridcolor="#1e3050"),
            height=350,
        )

        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Please select at least 1 feature from the sidebar for comparison.")

    # ── Silhouette Reference ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Silhouette Score Reference</div>", unsafe_allow_html=True)

    ref_data = {
        "Range": ["0.71 – 1.00", "0.51 – 0.70", "0.26 – 0.50", "< 0.26"],
        "Quality": ["Excellent", "Good", "Fair", "Poor"],
        "Meaning": [
            "Dense, well-separated clusters",
            "Reasonable cluster structure",
            "Weak or overlapping clusters",
            "Clusters may not be meaningful",
        ],
    }

    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ═══ TAB 2: K-FOLD CROSS-VALIDATION (NEW) ════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>K-Fold Cross-Validation</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <p style='color:#7a96b8'>
    K-Fold validation splits the data into K equal parts. The model is trained on K-1 parts
    and tested on the remaining part — repeated K times. This gives a reliable estimate of
    model performance and reduces overfitting bias compared to a single train/test split.
    </p>
    </div>
    """, unsafe_allow_html=True)

    cv_col1, cv_col2, cv_col3 = st.columns(3)
    with cv_col1:
        k_folds = st.slider("Number of Folds (K)", min_value=2, max_value=10, value=5, key="k_folds",
                            help="More folds = more reliable estimate but slower")
    with cv_col2:
        cv_model_type = st.selectbox("Model for CV", ["Random Forest", "SVM"], key="cv_model")
    with cv_col3:
        cv_k_clusters = st.slider("K-Means clusters (target labels)", 2, 8, 5, key="cv_k_clusters")

    if st.button("▶️ Run K-Fold Validation", key="run_kfold"):
        with st.spinner(f"Running {k_folds}-Fold cross-validation..."):
            # Generate cluster labels as classification target
            km_labels_cv, _ = run_kmeans(X_scaled, n_clusters=cv_k_clusters)
            y_cv = km_labels_cv

            if cv_model_type == "Random Forest":
                cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                cv_model = SVC(kernel="rbf", random_state=42)

            results = kfold_cross_validate(cv_model, X_scaled, y_cv, k=k_folds, scoring="accuracy")

            st.success(f"✅ {k_folds}-Fold CV completed!")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Accuracy", f"{results['mean']:.4f}")
            m2.metric("Std Dev", f"± {results['std']:.4f}")
            m3.metric("Folds", results['k'])
            m4.metric("Model", cv_model_type)

            # Per-fold bar chart
            fold_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(results["scores"]))],
                "Accuracy": results["scores"],
            })

            fig_cv = px.bar(
                fold_df, x="Fold", y="Accuracy",
                color="Accuracy",
                color_continuous_scale="Blues",
                text=[f"{s:.4f}" for s in results["scores"]],
                template="plotly_dark",
                title=f"{k_folds}-Fold Cross-Validation Accuracy per Fold",
            )
            fig_cv.add_hline(
                y=results["mean"],
                line_dash="dash",
                line_color="#00e5cc",
                annotation_text=f"Mean = {results['mean']:.4f}",
                annotation_font_color="#00e5cc",
            )
            fig_cv.update_traces(textposition="outside")
            fig_cv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
                title_font_color="#00e5cc", font_color="#e0e6f0",
                xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
                height=420,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"""
            <p style='color:#7a96b8'>
            📊 <strong style='color:#00e5cc'>Results Summary</strong><br>
            Model: <code>{cv_model_type}</code> &nbsp;|&nbsp;
            Folds: <code>{k_folds}</code> &nbsp;|&nbsp;
            Mean Accuracy: <code>{results['mean']:.4f}</code> &nbsp;±&nbsp; <code>{results['std']:.4f}</code><br>
            Individual fold scores: {[round(s, 4) for s in results['scores']]}
            </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Configure settings above and click **Run K-Fold Validation** to start.")


# ═══ TAB 3: HYPERPARAMETER TUNING (NEW) ══════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Hyperparameter Tuning (GridSearchCV)</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <p style='color:#7a96b8'>
    GridSearchCV exhaustively tries all parameter combinations using cross-validation
    to find the settings that produce the best model performance. Choose your model
    and click Tune to find the optimal hyperparameters.
    </p>
    </div>
    """, unsafe_allow_html=True)

    tune_col1, tune_col2 = st.columns(2)
    with tune_col1:
        tune_model = st.selectbox(
            "Model to Tune",
            ["KMeans (n_clusters)", "Random Forest (n_estimators, max_depth)", "SVM (kernel, C)"],
            key="tune_model"
        )
    with tune_col2:
        tune_k_target = st.slider("K-Means clusters (for target labels)", 2, 8, 5, key="tune_k_target")

    if st.button("🔬 Run Hyperparameter Tuning", key="run_tuning"):
        with st.spinner("Running GridSearchCV — this may take a moment..."):

            if "KMeans" in tune_model:
                # KMeans tuning via silhouette sweep
                results = tune_kmeans(X_scaled, n_clusters_range=list(range(2, 10)))

                st.success(f"✅ KMeans tuning complete! Best K = {results['best_k']}")

                m1, m2 = st.columns(2)
                m1.metric("Best n_clusters", results["best_k"])
                m2.metric("Best Silhouette Score", f"{results['best_score']:.4f}")

                scores_df = pd.DataFrame({
                    "n_clusters": list(results["scores_by_k"].keys()),
                    "Silhouette Score": list(results["scores_by_k"].values()),
                })

                fig_tune = px.bar(
                    scores_df, x="n_clusters", y="Silhouette Score",
                    color="Silhouette Score",
                    color_continuous_scale="Blues",
                    text=[f"{v:.4f}" for v in scores_df["Silhouette Score"]],
                    template="plotly_dark",
                    title="KMeans Silhouette Score by n_clusters",
                )
                fig_tune.add_vline(
                    x=results["best_k"],
                    line_dash="dash", line_color="#00e5cc",
                    annotation_text=f"Best K={results['best_k']}",
                    annotation_font_color="#00e5cc",
                )
                fig_tune.update_traces(textposition="outside")
                fig_tune.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
                    title_font_color="#00e5cc", font_color="#e0e6f0",
                    xaxis=dict(gridcolor="#1e3050"), yaxis=dict(gridcolor="#1e3050"),
                    height=420,
                )
                st.plotly_chart(fig_tune, use_container_width=True)

            else:
                # Generate classification target from KMeans labels
                km_labels_t, _ = run_kmeans(X_scaled, n_clusters=tune_k_target)
                y_tune = km_labels_t

                if "Random Forest" in tune_model:
                    results = tune_random_forest(X_scaled, y_tune, scoring="accuracy")
                    st.success("✅ Random Forest tuning complete!")

                    m1, m2 = st.columns(2)
                    m1.metric("Best Score (CV)", f"{results['best_score']:.4f}")
                    m2.metric("Best Params", str(results["best_params"]))

                    st.markdown("<div class='section-header'>GridSearch Results</div>", unsafe_allow_html=True)
                    st.dataframe(
                        results["cv_results"].rename(columns={
                            "param_n_estimators": "n_estimators",
                            "param_max_depth": "max_depth",
                            "mean_test_score": "Mean CV Score",
                            "std_test_score": "Std Dev",
                        }),
                        use_container_width=True
                    )

                    # Heatmap of scores
                    try:
                        pivot_df = results["cv_results"].copy()
                        pivot_df.columns = ["n_estimators", "max_depth", "Mean CV Score", "Std"]
                        pivot_df["max_depth"] = pivot_df["max_depth"].astype(str)
                        fig_heat = px.density_heatmap(
                            pivot_df, x="n_estimators", y="max_depth", z="Mean CV Score",
                            color_continuous_scale="Blues",
                            template="plotly_dark",
                            title="Random Forest GridSearch Score Heatmap",
                        )
                        fig_heat.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,21,32,0.8)",
                            title_font_color="#00e5cc", font_color="#e0e6f0",
                            height=360,
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                    except Exception:
                        pass

                else:  # SVM
                    results = tune_svm(X_scaled, y_tune)
                    st.success("✅ SVM tuning complete!")

                    m1, m2 = st.columns(2)
                    m1.metric("Best Score (CV)", f"{results['best_score']:.4f}")
                    m2.metric("Best Params", str(results["best_params"]))

                    st.markdown("<div class='section-header'>GridSearch Results</div>", unsafe_allow_html=True)
                    st.dataframe(
                        results["cv_results"].rename(columns={
                            "param_kernel": "kernel",
                            "param_C": "C",
                            "mean_test_score": "Mean CV Score",
                            "std_test_score": "Std Dev",
                        }),
                        use_container_width=True
                    )

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"""
                <p style='color:#7a96b8'>
                🏆 <strong style='color:#00e5cc'>Best Parameters Found:</strong><br>
                {' &nbsp;|&nbsp; '.join([f"<code>{k}</code> = <code>{v}</code>"
                                         for k, v in results['best_params'].items()])}<br>
                Best CV Score: <code>{results['best_score']:.4f}</code>
                </p>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Configure settings above and click **Run Hyperparameter Tuning** to start.")
