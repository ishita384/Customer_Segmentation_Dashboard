"""
outlier.py
----------
Outlier detection using Isolation Forest.
Isolation Forest works by randomly partitioning the data;
anomalies are isolated faster (shorter path length) than normal points.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


def detect_outliers(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """
    Fit Isolation Forest and return:
    - labels  : array of  1 (inlier) or -1 (outlier)
    - scores  : anomaly score per sample (lower = more anomalous)
    - mask    : boolean array — True for inliers

    contamination: expected fraction of outliers in the dataset.
    Typical values: 0.01 – 0.10 (1 % – 10 %).
    """
    # Fix NaN before Isolation Forest (critical)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,   # number of trees — more trees → more stable
    )
    labels = iso.fit_predict(X)          # 1 = inlier, -1 = outlier
    scores = iso.decision_function(X)    # higher score = more normal
    mask = labels == 1                   # boolean mask for inliers
    return labels, scores, mask


def outlier_summary(labels: np.ndarray) -> dict:
    """Return counts of inliers and outliers."""
    n_outliers = int(np.sum(labels == -1))
    n_inliers  = int(np.sum(labels == 1))
    return {
        "n_outliers": n_outliers,
        "n_inliers":  n_inliers,
        "pct_outliers": round(100 * n_outliers / len(labels), 2),
    }


def remove_outliers(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the cleaned dataset with outlier rows removed."""
    return X[mask]
