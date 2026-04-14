"""
Customer Segmentation Dashboard
Main entry point — Landing page + ML Pipeline (merged).
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (unchanged from original) ────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117; color: #e0e6f0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0e1117 100%);
    border-right: 1px solid #1e2d45;
}
.card {
    background: linear-gradient(135deg, #141c2e 0%, #0e1520 100%);
    border: 1px solid #1e3050; border-radius: 16px;
    padding: 24px 28px; margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,180,255,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,180,255,0.14); }
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #00b4ff, #00e5cc, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2; margin-bottom: 0.3rem;
}
.hero-sub { font-size: 1.05rem; color: #7a96b8; margin-top: 0; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #141c2e, #0e1520);
    border: 1px solid #1e3050; border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #00e5cc; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #7a96b8; }
.stButton > button {
    background: linear-gradient(135deg, #0070f3, #00b4d8);
    color: white; border: none; border-radius: 10px;
    padding: 0.5rem 1.4rem; font-weight: 600; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
[data-testid="stTabs"] [data-baseweb="tab"] { color: #7a96b8; border-radius: 8px 8px 0 0; }
[data-testid="stTabs"] [aria-selected="true"] {
    color: #00e5cc !important; border-bottom: 2px solid #00e5cc !important;
}
.stSlider [data-baseweb="slider"] { color: #00b4ff; }
div[data-baseweb="select"] > div { background-color: #141c2e; border-color: #1e3050; }
.section-header {
    font-size: 1.25rem; font-weight: 700; color: #00b4ff;
    border-left: 4px solid #00e5cc; padding-left: 12px; margin: 24px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════
tab_home, tab_pipeline = st.tabs(["🏠 Home", "🔀 ML Pipeline"])


# ════════════════════════════════════════════════════════
# TAB 1 — HOME
# ════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class='card'>
        <div class='hero-title'>🧬 Customer Segmentation Dashboard</div>
        <p class='hero-sub'>An end-to-end Machine Learning pipeline — from raw data to actionable clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("ML Techniques", "7+", "KMeans · DBSCAN · RF · SVM · PCA · IsoForest")
    with col2: st.metric("Pages", "7", "Full pipeline coverage")
    with col3: st.metric("Dataset", "Any CSV", "Default: Mall Customers 200 records")
    with col4: st.metric("Evaluation", "Silhouette + K-Fold", "Industry standard")

    st.markdown("<div class='section-header'>🗺️ Navigation Guide</div>", unsafe_allow_html=True)

    pages = [
        ("📊", "Data Explorer",     "Browse raw data, distributions & correlations"),
        ("⚙️", "Preprocessing",     "Cleaning, encoding, scaling + Feature Selection"),
        ("🧠", "Clustering",        "KMeans (Elbow) + DBSCAN + RF / SVM / LinearReg"),
        ("📉", "PCA Visualization", "Reduce to 2D and explore cluster geometry"),
        ("⚠️", "Outliers",          "Isolation Forest — detect & optionally remove"),
        ("📈", "Evaluation",        "Silhouette · K-Fold CV · Hyperparameter Tuning"),
        ("💡", "Insights",          "Business recommendations from cluster profiles"),
    ]

    page_cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(pages):
        with page_cols[i % 2]:
            st.markdown(f"""
            <div class='card'>
                <span style='font-size:1.6rem'>{icon}</span>
                <strong style='color:#00e5cc; font-size:1.05rem'> {title}</strong>
                <p style='color:#7a96b8; margin:6px 0 0 0; font-size:0.9rem'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='border-color:#7c3aed44; text-align:center; padding:18px;'>
        <span style='color:#7a96b8; font-size:0.9rem'>
            👈 Use the <strong style='color:#00e5cc'>sidebar</strong> to navigate &nbsp;|&nbsp;
            Built with <strong style='color:#00b4ff'>Python · Streamlit · Scikit-learn · Plotly</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 — ML PIPELINE
# ════════════════════════════════════════════════════════
with tab_pipeline:

    st.markdown("""
    <div class='card'>
        <div class='hero-title'>🔀 ML Pipeline</div>
        <p class='hero-sub'>10-step pipeline — see every stage, then run them all at once.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── FLOWCHART via st.columns (no f-string HTML injection) ────────────────
    st.markdown("<div class='section-header'>Pipeline Flowchart</div>", unsafe_allow_html=True)

    STEPS = [
        ("📥", "Input\nData",   "1"),
        ("🔍", "EDA",           "2"),
        ("🧹", "Cleaning",      "3"),
        ("🎯", "Feature\nSel.", "4"),
        ("✂️", "Data\nSplit",   "5"),
        ("🤖", "Model\nSelect", "6"),
        ("🏋️", "Training",     "7"),
        ("🔁", "K-Fold\nCV",    "8"),
        ("📊", "Metrics",       "9"),
        ("⚙️", "HP\nTuning",   "10"),
    ]

    BOX_STYLE = (
        "background:linear-gradient(135deg,#141c2e,#0e1520);"
        "border:1.5px solid #1e3050;border-radius:12px;"
        "padding:10px 4px;text-align:center;min-height:88px;"
    )
    NUM_STYLE  = "font-size:0.55rem;color:#7a96b8;font-weight:700;text-transform:uppercase;margin-bottom:3px"
    ICON_STYLE = "font-size:1.4rem;line-height:1"
    LBL_STYLE  = "font-size:0.60rem;color:#e0e6f0;font-weight:600;margin-top:4px;line-height:1.3"

    step_cols = st.columns(len(STEPS))
    for i, (icon, label, num) in enumerate(STEPS):
        label_html = label.replace("\n", "<br>")
        with step_cols[i]:
            st.markdown(
                f"<div style='{BOX_STYLE}'>"
                f"<div style='{NUM_STYLE}'>Step {num}</div>"
                f"<div style='{ICON_STYLE}'>{icon}</div>"
                f"<div style='{LBL_STYLE}'>{label_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Connector arrows on a second row
    arr_cols = st.columns(len(STEPS))
    for i in range(len(STEPS) - 1):
        with arr_cols[i]:
            st.markdown(
                "<div style='text-align:right;color:#00b4ff;font-size:1rem;"
                "margin-top:2px;opacity:0.5'>→</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step description table (collapsed) ───────────────────────────────────
    with st.expander("📖 Step Descriptions", expanded=False):
        step_desc = pd.DataFrame({
            "Step": [s[2] for s in STEPS],
            "Stage": ["Input Data","EDA","Data Cleaning","Feature Selection","Data Split",
                      "Model Selection","Model Training","K-Fold Validation","Performance Metrics","HP Tuning"],
            "What It Does": [
                "Load CSV or use built-in sample — stored in session state for all pages.",
                "Distributions, correlations, missing values, data types.",
                "Median fill, label encoding, feature engineering (TotalValue, AgeGroup).",
                "Variance Threshold + Correlation Filter drop low-value / redundant features.",
                "80/20 train-test split via sklearn — prevents data leakage.",
                "KMeans, DBSCAN, Random Forest, SVM, or Linear Regression.",
                "Fit on training data, evaluate on held-out test set.",
                "K-Fold CV for reliable, unbiased performance estimate across K folds.",
                "Silhouette Score / Accuracy / R² / MSE depending on model type.",
                "GridSearchCV over n_clusters / n_estimators / kernel / C.",
            ],
        })
        st.dataframe(step_desc, use_container_width=True, hide_index=True)

    # ── RUNNER ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>▶️ Run Full Pipeline</div>", unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ Go to **Data Explorer** (sidebar) and load a dataset first.")
        st.stop()

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    from preprocessing import full_pipeline, apply_variance_threshold, apply_correlation_filter, scale_features
    from clustering import run_kmeans, compute_silhouette
    from evaluation import tune_kmeans

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        pipe_model = st.selectbox(
            "Model",
            ["KMeans (Clustering)", "Random Forest (Classifier)", "SVM Classifier", "Linear Regression"],
            key="pipe_model",
        )
    with cfg2:
        pipe_k     = st.slider("KMeans n_clusters", 2, 10, 5, key="pipe_k")
        pipe_kfold = st.slider("K-Fold splits", 2, 10, 5, key="pipe_kfold")
    with cfg3:
        pipe_test  = st.slider("Test split (%)", 10, 40, 20, key="pipe_test") / 100
        pipe_var   = st.slider("Variance Threshold", 0.0, 0.5, 0.01, 0.01, key="pipe_var")

    pipe_corr = st.slider("Correlation Filter Threshold", 0.5, 1.0, 0.95, 0.01, key="pipe_corr")

    if st.button("🚀 Run Full Pipeline", key="run_pipeline"):

        imputer   = SimpleImputer(strategy="mean")
        log_items = []   # (text, colour)

        def log(msg, color="#e0e6f0"):
            log_items.append((msg, color))

        prog   = st.progress(0)
        status = st.empty()

        # Step 1
        status.markdown("**Step 1/10** — Input Data")
        log("━" * 48, "#1e3050")
        log("📥  STEP 1 — Input Data", "#00b4ff")
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        cc = df.select_dtypes(include="object").columns.tolist()
        log(f"   Shape: {df.shape[0]} rows × {df.shape[1]} cols   Numeric: {len(nc)}   Cat: {len(cc)}", "#00e5cc")
        prog.progress(10)

        # Step 2
        status.markdown("**Step 2/10** — EDA")
        log("━" * 48, "#1e3050")
        log("🔍  STEP 2 — EDA", "#00b4ff")
        log(f"   Missing values: {int(df.isnull().sum().sum())}   Columns: {list(df.columns)}", "#00e5cc")
        prog.progress(20)

        # Step 3
        status.markdown("**Step 3/10** — Data Cleaning")
        log("━" * 48, "#1e3050")
        log("🧹  STEP 3 — Data Engineering & Cleaning", "#00b4ff")
        pr = full_pipeline(df)
        X_scaled     = imputer.fit_transform(pr["X_scaled"])
        df_feat      = pr["featured"]
        feature_cols = pr["feature_cols"]
        log(f"   NaN filled (median), categoricals encoded, features engineered", "#00e5cc")
        log(f"   Feature matrix: {X_scaled.shape}", "#00e5cc")
        prog.progress(30)

        # Step 4
        status.markdown("**Step 4/10** — Feature Selection")
        log("━" * 48, "#1e3050")
        log("🎯  STEP 4 — Feature Selection", "#00b4ff")
        after_var  = apply_variance_threshold(df_feat, feature_cols, threshold=pipe_var)
        after_corr = apply_correlation_filter(df_feat, after_var, corr_threshold=pipe_corr)
        log(f"   {len(feature_cols)} → variance filter → {len(after_var)} → corr filter → {len(after_corr)}", "#00e5cc")
        log(f"   Selected: {after_corr}", "#00e5cc")
        X_sel, _ = scale_features(df_feat, after_corr)
        X_sel     = imputer.fit_transform(X_sel)
        prog.progress(40)

        # Step 5
        status.markdown("**Step 5/10** — Data Split")
        log("━" * 48, "#1e3050")
        log("✂️  STEP 5 — Data Split", "#00b4ff")
        X_train, X_test = train_test_split(X_sel, test_size=pipe_test, random_state=42)
        log(f"   Train: {len(X_train)} ({int((1-pipe_test)*100)}%)   Test: {len(X_test)} ({int(pipe_test*100)}%)", "#00e5cc")
        prog.progress(50)

        # Step 6
        status.markdown("**Step 6/10** — Model Selection")
        log("━" * 48, "#1e3050")
        log("🤖  STEP 6 — Model Selection", "#00b4ff")
        log(f"   Model: {pipe_model}", "#00e5cc")
        is_clustering = "KMeans" in pipe_model
        is_regression = "Regression" in pipe_model
        prog.progress(55)

        # Step 7
        status.markdown("**Step 7/10** — Training")
        log("━" * 48, "#1e3050")
        log("🏋️  STEP 7 — Model Training", "#00b4ff")

        trained_model = None
        y_labels      = None
        metric_name   = ""
        metric_val    = None
        mse_val       = None

        if is_clustering:
            km_labels, km_model = run_kmeans(X_sel, n_clusters=pipe_k)
            sil           = compute_silhouette(X_sel, km_labels)
            metric_name   = "Silhouette"
            metric_val    = sil
            y_labels      = km_labels
            trained_model = km_model
            log(f"   KMeans  K={pipe_k}  Inertia={km_model.inertia_:.2f}  Sil={sil:.4f}", "#00e5cc")
        else:
            km_labels_t, _ = run_kmeans(X_sel, n_clusters=pipe_k)
            y_labels = km_labels_t
            X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_labels, test_size=pipe_test, random_state=42)
            if is_regression:
                mdl = LinearRegression()
                mdl.fit(X_tr, y_tr.astype(float))
                preds     = mdl.predict(X_te)
                metric_name = "R²"
                metric_val  = r2_score(y_te, preds)
                mse_val     = mean_squared_error(y_te, preds)
                log(f"   Linear Regression  R²={metric_val:.4f}  MSE={mse_val:.4f}", "#00e5cc")
            elif "Random Forest" in pipe_model:
                mdl = RandomForestClassifier(n_estimators=100, random_state=42)
                mdl.fit(X_tr, y_tr)
                metric_name = "Accuracy"
                metric_val  = accuracy_score(y_te, mdl.predict(X_te))
                log(f"   Random Forest  Accuracy={metric_val:.4f}", "#00e5cc")
            else:
                mdl = SVC(kernel="rbf", random_state=42)
                mdl.fit(X_tr, y_tr)
                metric_name = "Accuracy"
                metric_val  = accuracy_score(y_te, mdl.predict(X_te))
                log(f"   SVM (rbf)  Accuracy={metric_val:.4f}", "#00e5cc")
            trained_model = mdl
        prog.progress(65)

        # Step 8
        status.markdown("**Step 8/10** — K-Fold Validation")
        log("━" * 48, "#1e3050")
        log("🔁  STEP 8 — K-Fold Cross-Validation", "#00b4ff")
        kfold_mean = kfold_std = None
        if not is_clustering:
            scoring    = "r2" if is_regression else "accuracy"
            cv_scores  = cross_val_score(trained_model, X_sel, y_labels, cv=pipe_kfold, scoring=scoring)
            kfold_mean = float(cv_scores.mean())
            kfold_std  = float(cv_scores.std())
            log(f"   {pipe_kfold}-Fold CV ({scoring}) = {kfold_mean:.4f} ± {kfold_std:.4f}", "#00e5cc")
            log(f"   Folds: {[round(s,4) for s in cv_scores.tolist()]}", "#00e5cc")
        else:
            log("   Skipped — not applicable to unsupervised clustering", "#ff9900")
        prog.progress(80)

        # Step 9
        status.markdown("**Step 9/10** — Metrics")
        log("━" * 48, "#1e3050")
        log("📊  STEP 9 — Performance Metrics", "#00b4ff")
        if metric_val is not None:
            log(f"   {metric_name}: {metric_val:.4f}", "#00e5cc")
        if kfold_mean is not None:
            log(f"   K-Fold mean: {kfold_mean:.4f} ± {kfold_std:.4f}", "#00e5cc")
        if mse_val is not None:
            log(f"   MSE: {mse_val:.4f}", "#00e5cc")
        log(f"   Features used: {len(after_corr)}  |  Samples: {len(X_sel)}", "#00e5cc")
        prog.progress(90)

        # Step 10
        status.markdown("**Step 10/10** — Hyperparameter Tuning")
        log("━" * 48, "#1e3050")
        log("⚙️  STEP 10 — Hyperparameter Tuning", "#00b4ff")

        best_param_str = best_score_str = ""
        tuning_rows = []

        if is_clustering:
            tr = tune_kmeans(X_sel, n_clusters_range=list(range(2, 9)))
            best_param_str = f"n_clusters={tr['best_k']}"
            best_score_str = f"{tr['best_score']:.4f} (silhouette)"
            log(f"   Best K={tr['best_k']}  Silhouette={tr['best_score']:.4f}", "#00e5cc")
            tuning_rows = [{"K": k, "Silhouette": v} for k, v in tr["scores_by_k"].items()]
        else:
            sc = "r2" if is_regression else "accuracy"
            if "Random Forest" in pipe_model:
                pg  = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
                gm  = RandomForestRegressor(random_state=42) if is_regression \
                      else RandomForestClassifier(random_state=42)
            elif is_regression:
                pg  = {"alpha": [0.01, 0.1, 1.0, 10.0]}
                gm  = Ridge()
            else:
                pg  = {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1.0, 10.0]}
                gm  = SVC(random_state=42)

            gs = GridSearchCV(gm, pg, cv=3, scoring=sc, n_jobs=-1)
            gs.fit(X_sel, y_labels)
            best_param_str = str(gs.best_params_)
            best_score_str = f"{gs.best_score_:.4f}"
            log(f"   Best params: {gs.best_params_}  Score: {gs.best_score_:.4f}", "#00e5cc")
            res_df     = pd.DataFrame(gs.cv_results_)
            param_cols = [c for c in res_df.columns if c.startswith("param_")]
            tuning_rows = (res_df[param_cols + ["mean_test_score", "std_test_score"]]
                           .round(4).to_dict("records"))

        log("━" * 48, "#1e3050")
        log("✅  PIPELINE COMPLETE", "#00e5cc")
        prog.progress(100)
        status.empty()

        # ── LOG ───────────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Pipeline Log</div>", unsafe_allow_html=True)
        log_html = "<br>".join(
            f"<span style='color:{c}'>{m.replace('<','&lt;').replace('>','&gt;')}</span>"
            for m, c in log_items
        )
        st.markdown(
            "<div style='background:#080d14;border:1px solid #1e3050;border-radius:12px;"
            "padding:16px 20px;font-family:monospace;font-size:0.82rem;"
            "max-height:340px;overflow-y:auto;line-height:1.8'>"
            + log_html + "</div>",
            unsafe_allow_html=True,
        )

        # ── RESULTS ───────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Results</div>", unsafe_allow_html=True)
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Model",            pipe_model.split(" ")[0])
        r2.metric("Features",         len(after_corr))
        r3.metric(metric_name,        f"{metric_val:.4f}" if metric_val is not None else "N/A")
        if kfold_mean is not None:
            r4.metric(f"{pipe_kfold}-Fold Mean", f"{kfold_mean:.4f}")
            r5.metric("CV Std Dev",   f"±{kfold_std:.4f}")
        else:
            r4.metric("Inertia",      f"{trained_model.inertia_:,.0f}" if is_clustering else "—")
            r5.metric("Best Score",   best_score_str[:14])

        if tuning_rows:
            st.markdown("<div class='section-header'>⚙️ Tuning Results</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(tuning_rows), use_container_width=True, hide_index=True)

        kf_line = (f"K-Fold Mean: **{kfold_mean:.4f} ± {kfold_std:.4f}** | "
                   if kfold_mean is not None else "")
        st.success(
            f"✅ Complete! Model: **{pipe_model}** | Features: **{len(after_corr)}** | "
            f"{metric_name}: **{metric_val:.4f}** | {kf_line}"
            f"Best params: `{best_param_str}`"
        )

    else:
        st.info("👆 Configure settings above and click **Run Full Pipeline**.")
