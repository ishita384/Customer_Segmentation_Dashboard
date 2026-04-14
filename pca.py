"""
pca.py
------
Principal Component Analysis — used ONLY for 2-D visualization.
PCA finds the directions of maximum variance in high-dimensional data
and projects data onto those directions (principal components).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def apply_pca(X: np.ndarray, n_components: int = 2) -> tuple:
    """
    Reduce X to `n_components` dimensions using PCA.
    Includes NaN imputation before PCA to prevent errors.
    Returns:
    - X_pca      : transformed array (n_samples, n_components)
    - pca        : fitted PCA object (has .explained_variance_ratio_)
    - explained  : % variance explained by each component
    """
    # Fix NaN before PCA (critical)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Clamp n_components to valid range
    n_components = min(n_components, X.shape[1], X.shape[0])

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Explained variance as percentages
    explained = (pca.explained_variance_ratio_ * 100).round(2).tolist()
    return X_pca, pca, explained


def pca_summary(pca: PCA) -> dict:
    """
    Summarise PCA output for display:
    - explained_variance_ratio : list of % variance per component
    - total_explained          : cumulative variance of all chosen components
    """
    ratios = (pca.explained_variance_ratio_ * 100).round(2).tolist()
    return {
        "explained_variance_ratio": ratios,
        "total_explained": round(sum(ratios), 2),
        "n_components": pca.n_components_,
    }
