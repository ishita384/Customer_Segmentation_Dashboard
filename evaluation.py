"""
evaluation.py
-------------
Clustering evaluation using Silhouette Score.
Also provides K-Fold cross-validation and GridSearchCV hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer


def silhouette_per_sample(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute per-sample silhouette scores.
    Useful for visualising which points are well-classified.
    """
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return np.zeros(len(labels))
    scores = silhouette_samples(X[mask], labels[mask])
    # Pad noise points with 0
    full = np.zeros(len(labels))
    full[mask] = scores
    return full


def compare_feature_sets(
    X_all: np.ndarray,
    X_selected: np.ndarray,
    n_clusters: int = 5,
) -> dict:
    """
    Run K-Means on two feature sets and compare Silhouette Scores.
    - X_all      : all features
    - X_selected : a subset of important features
    Returns dict with scores for both sets.
    """
    results = {}
    for label, X in [("All Features", X_all), ("Selected Features", X_selected)]:
        # Impute NaNs
        imputer = SimpleImputer(strategy="mean")
        X_clean = imputer.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labs = km.fit_predict(X_clean)
        score = silhouette_score(X_clean, labs) if len(set(labs)) > 1 else float("nan")
        results[label] = {
            "silhouette": round(float(score), 4),
            "n_clusters": n_clusters,
            "inertia": round(float(km.inertia_), 2),
        }
    return results


def interpret_silhouette(score: float) -> str:
    """Return a human-readable interpretation of the silhouette score."""
    if np.isnan(score):
        return "Cannot compute (need ≥ 2 clusters)"
    if score >= 0.71:
        return "🟢 Excellent — clusters are dense and well separated"
    if score >= 0.51:
        return "🟡 Good — reasonable cluster structure"
    if score >= 0.26:
        return "🟠 Fair — weak or overlapping clusters"
    return "🔴 Poor — clusters may not be meaningful"


# ── K-Fold Cross-Validation ──────────────────────────────────────────────────

def kfold_cross_validate(model, X: np.ndarray, y: np.ndarray, k: int = 5, scoring: str = "accuracy") -> dict:
    """
    Run K-Fold cross-validation on a supervised model.
    
    Parameters:
    - model   : a scikit-learn estimator (e.g. RandomForestClassifier)
    - X       : feature matrix
    - y       : target vector
    - k       : number of folds (default 5)
    - scoring : metric to use ('accuracy', 'r2', 'neg_mean_squared_error')
    
    Returns a dict with mean, std, and all fold scores.
    """
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(X)

    scores = cross_val_score(model, X_clean, y, cv=k, scoring=scoring)
    return {
        "scores": scores.tolist(),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "k": k,
        "scoring": scoring,
    }


# ── Hyperparameter Tuning ────────────────────────────────────────────────────

def tune_kmeans(X: np.ndarray, n_clusters_range: list = None) -> dict:
    """
    Tune KMeans by trying different n_clusters values.
    Returns the best n_clusters based on silhouette score.
    """
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(X)

    if n_clusters_range is None:
        n_clusters_range = list(range(2, 9))

    results = {}
    best_score = -1
    best_k = n_clusters_range[0]

    for k in n_clusters_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_clean)
        if len(set(labels)) >= 2:
            score = silhouette_score(X_clean, labels)
            results[k] = round(float(score), 4)
            if score > best_score:
                best_score = score
                best_k = k
        else:
            results[k] = float("nan")

    return {
        "scores_by_k": results,
        "best_k": best_k,
        "best_score": round(best_score, 4),
    }


def tune_random_forest(X: np.ndarray, y: np.ndarray, scoring: str = "accuracy") -> dict:
    """
    Hyperparameter tuning for Random Forest using GridSearchCV.
    Tunes n_estimators and max_depth.
    Returns best params and best score.
    """
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(X)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    }

    if scoring == "accuracy":
        model = RandomForestClassifier(random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        if scoring == "accuracy":
            scoring = "r2"

    grid = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
    grid.fit(X_clean, y)

    return {
        "best_params": grid.best_params_,
        "best_score": round(float(grid.best_score_), 4),
        "cv_results": pd.DataFrame(grid.cv_results_)[
            ["param_n_estimators", "param_max_depth", "mean_test_score", "std_test_score"]
        ].round(4),
    }


def tune_svm(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Hyperparameter tuning for SVM using GridSearchCV.
    Tunes kernel and C (regularisation strength).
    Returns best params and best score.
    """
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(X)

    param_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1.0, 10.0],
    }

    model = SVC(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_clean, y)

    return {
        "best_params": grid.best_params_,
        "best_score": round(float(grid.best_score_), 4),
        "cv_results": pd.DataFrame(grid.cv_results_)[
            ["param_kernel", "param_C", "mean_test_score", "std_test_score"]
        ].round(4),
    }
