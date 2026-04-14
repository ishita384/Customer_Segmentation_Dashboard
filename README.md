# 🧬 Customer Segmentation Dashboard — Enhanced ML Pipeline

A complete, end-to-end Machine Learning pipeline dashboard built with Streamlit.
Works with **any CSV dataset** — not just Mall Customers.

---

## 📁 Project Structure

```
customer_segmentation/
│
├── app.py                  ← Landing page + navigation
├── preprocessing.py        ← Data loading, cleaning, encoding, feature selection
├── clustering.py           ← KMeans, DBSCAN, RF, SVM, Linear Regression, data split
├── pca.py                  ← PCA dimensionality reduction (with NaN fix)
├── evaluation.py           ← Silhouette, K-Fold CV, GridSearchCV tuning
├── outlier.py              ← Isolation Forest (with NaN fix)
├── requirements.txt
│
└── pages/
    ├── 1_Data_Explorer.py      ← EDA: upload or use sample dataset
    ├── 2_Preprocessing.py      ← Cleaning + Feature Selection (NEW)
    ├── 3_Clustering.py         ← KMeans + DBSCAN + Supervised Models (NEW)
    ├── 4_PCA_Visualization.py  ← PCA 2D scatter + loadings
    ├── 5_Outliers.py           ← Isolation Forest
    ├── 6_Evaluation.py         ← Silhouette + K-Fold CV + Hyperparameter Tuning (NEW)
    ├── 7_Insights.py           ← Business segment recommendations
    └── 8_ML_Pipeline.py        ← Full 10-step pipeline flowchart + runner (NEW)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

Streamlit will print a URL like:
```
Local URL: http://localhost:8501
```

Open that URL in your browser.

---

## 🔥 New Features Added

| Feature | Location |
|---|---|
| **Session state data flow** | All pages — `st.session_state["df"]` |
| **Dynamic column detection** | All pages — no hardcoded column names |
| **NaN fix (SimpleImputer)** | preprocessing.py, clustering.py, pca.py, outlier.py |
| **Variance Threshold feature selection** | 2_Preprocessing.py |
| **Correlation filter feature selection** | 2_Preprocessing.py |
| **Train / Test split** | 3_Clustering.py |
| **Random Forest model** | 3_Clustering.py |
| **SVM model** | 3_Clustering.py |
| **Linear Regression model** | 3_Clustering.py |
| **K-Fold cross-validation** | 6_Evaluation.py |
| **GridSearchCV hyperparameter tuning** | 6_Evaluation.py |
| **Performance metrics (Accuracy, R², MSE)** | 6_Evaluation.py |
| **Full pipeline flowchart + runner** | 8_ML_Pipeline.py |

---

## 📊 Pipeline Steps

1. **Input Data** — Load any CSV or use sample data
2. **EDA** — Distributions, correlations, missing values
3. **Data Cleaning** — Missing value fill, encoding, feature engineering
4. **Feature Selection** — Variance threshold + correlation filter
5. **Data Split** — 80/20 train/test split
6. **Model Selection** — KMeans, DBSCAN, RF, SVM, LinearReg
7. **Model Training** — Fit and evaluate on test set
8. **K-Fold Validation** — Reliable CV score across K folds
9. **Performance Metrics** — Silhouette / Accuracy / R² / MSE
10. **Hyperparameter Tuning** — GridSearchCV over key parameters

---

## 💡 Tips

- **Start with Data Explorer** — this stores your dataset in session state
- All other pages depend on session state; visit Data Explorer first
- Upload any numeric CSV to use a custom dataset
- Preprocessing page saves `selected_feature_cols` to session state for downstream use
- The ML Pipeline page runs all 10 steps end-to-end with one click
