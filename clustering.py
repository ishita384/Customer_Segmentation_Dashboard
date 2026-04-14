"""
clustering.py
-------------
K-Means, DBSCAN clustering plus Random Forest, SVM, and Linear Regression models.
Includes train/test split and silhouette evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, accuracy_score, r2_score,
    mean_squared_error, classification_report
)
from sklearn.impute import SimpleImputer


# ── K-Means ──────────────────────────────────────────────────────────────────

def compute_elbow(X: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Compute Within-Cluster Sum of Squares (WCSS / inertia) for each k.
    The 'elbow' point is where adding more clusters gives diminishing returns.
    Returns a dict: {k: inertia}
    """
    # Safety: impute NaNs first
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias[k] = km.inertia_
    return inertias


def run_kmeans(X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Fit K-Means with the given number of clusters.
    n_init=10 → run 10 random initialisations, pick the best.
    Returns cluster label array.
    """
    # Safety: impute NaNs first
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


# ── DBSCAN ───────────────────────────────────────────────────────────────────

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Fit DBSCAN.
    - eps        : max distance between two points to be neighbours
    - min_samples: minimum points in a neighbourhood to form a core point
    - Label -1   : noise / outlier points (not assigned to any cluster)
    Returns label array (noise = -1).
    """
    # Safety: impute NaNs first
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels


def dbscan_summary(labels: np.ndarray) -> dict:
    """
    Summarise DBSCAN output:
    - n_clusters : number of real clusters (exclude noise label -1)
    - n_noise    : number of noise points
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return {"n_clusters": n_clusters, "n_noise": n_noise}


# ── Silhouette ────────────────────────────────────────────────────────────────

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Silhouette Score.
    Range: [-1, 1]   Higher is better.
    Measures how similar each point is to its own cluster vs other clusters.
    Requires at least 2 clusters and labels not all noise.
    """
    unique = set(labels)
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return float("nan")
    try:
        score = silhouette_score(X[mask], labels[mask])
        return round(float(score), 4)
    except Exception:
        return float("nan")


# ── Train / Test Split ────────────────────────────────────────────────────────

def split_data(X: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix into train and test sets.
    Returns X_train, X_test.
    For clustering use cases we split on features only (no target).
    test_size: fraction for test set (default 0.2 = 20%)
    """
    # Safety: impute NaNs first
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    return X_train, X_test


def split_data_supervised(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target for supervised models.
    Returns X_train, X_test, y_train, y_test.
    """
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ── Multi-Model Selection ────────────────────────────────────────────────────

def run_random_forest_classifier(X_train, X_test, y_train, y_test, n_estimators=100):
    """
    Train a Random Forest Classifier.
    Returns model, accuracy, and classification report.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return model, acc, report


def run_svm_classifier(X_train, X_test, y_train, y_test, kernel="rbf"):
    """
    Train a Support Vector Machine Classifier.
    Returns model and accuracy.
    """
    model = SVC(kernel=kernel, random_state=42, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc


def run_linear_regression(X_train, X_test, y_train, y_test):
    """
    Train a Linear Regression model.
    Returns model, R² score, and MSE.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return model, round(r2, 4), round(mse, 4)


def run_random_forest_regressor(X_train, X_test, y_train, y_test, n_estimators=100):
    """
    Train a Random Forest Regressor.
    Returns model, R² score, and MSE.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return model, round(r2, 4), round(mse, 4)


# ── Live Cluster Prediction ───────────────────────────────────────────────────

def predict_cluster(model, scaler, input_dict: dict, feature_cols: list) -> int:
    """
    Predict the cluster for a single new customer.

    Parameters
    ----------
    model        : trained classifier (Random Forest or SVM)
    scaler       : fitted StandardScaler from the pipeline
    input_dict   : {feature_name: value} for each feature the user typed
    feature_cols : ordered list of feature names the model was trained on

    Returns
    -------
    cluster_id : int — predicted cluster label
    proba      : dict {cluster_label: probability} if model supports predict_proba
    """
    # Build a 1-row DataFrame in the same column order as training
    row = pd.DataFrame([input_dict], columns=feature_cols)

    # Impute any missing values (safety net)
    imputer = SimpleImputer(strategy="mean")
    # fit on a dummy 2-row array so imputer doesn't complain about 1 row
    row_arr = row.values.astype(float)

    # Scale using the pipeline scaler
    row_scaled = scaler.transform(row_arr)

    # Predict
    cluster_id = int(model.predict(row_scaled)[0])

    # Probability (not all models have it)
    proba = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row_scaled)[0]
        classes = model.classes_
        proba = {int(c): round(float(p), 4) for c, p in zip(classes, probs)}

    return cluster_id, proba
