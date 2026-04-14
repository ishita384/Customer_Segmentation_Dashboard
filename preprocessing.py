"""
preprocessing.py
----------------
Functions for data loading, cleaning, encoding, scaling, and feature selection.
Kept simple and beginner-friendly.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from io import StringIO


# ── Sample dataset (Mall Customers) ─────────────────────────────────────────
SAMPLE_CSV = """CustomerID,Gender,Age,AnnualIncome,SpendingScore
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Male,29,28,82
27,Female,45,28,32
28,Male,35,28,61
29,Female,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Female,21,30,73
33,Male,53,33,4
34,Male,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Female,42,34,17
38,Female,30,34,73
39,Male,36,37,26
40,Female,20,37,75
41,Male,65,38,35
42,Male,24,38,92
43,Female,48,39,36
44,Female,31,39,61
45,Female,49,39,28
46,Female,24,39,65
47,Female,50,40,55
48,Female,27,40,47
49,Female,29,40,42
50,Male,31,40,42
51,Male,49,42,52
52,Female,33,42,60
53,Female,31,43,54
54,Male,59,43,60
55,Female,50,43,45
56,Male,47,43,41
57,Female,51,44,50
58,Male,69,44,46
59,Female,27,46,51
60,Male,53,46,46
61,Male,70,46,56
62,Female,19,46,55
63,Female,67,47,52
64,Female,54,47,59
65,Male,63,48,51
66,Male,18,48,59
67,Female,43,48,50
68,Female,68,48,48
69,Male,19,48,59
70,Female,32,48,47
71,Male,70,49,55
72,Female,47,49,42
73,Female,60,50,49
74,Female,60,50,56
75,Male,59,54,47
76,Male,26,54,54
77,Female,45,54,53
78,Male,40,54,48
79,Female,23,54,52
80,Male,49,54,42
81,Female,57,54,51
82,Male,38,54,55
83,Female,67,54,41
84,Male,46,54,44
85,Female,21,54,57
86,Male,48,54,46
87,Female,55,57,58
88,Female,22,57,55
89,Female,34,58,60
90,Male,50,58,46
91,Female,68,59,55
92,Female,18,59,41
93,Male,48,60,49
94,Female,40,60,40
95,Male,32,60,42
96,Female,24,60,52
97,Male,47,60,47
98,Female,27,60,50
99,Male,48,61,42
100,Male,20,61,49
101,Female,23,62,41
102,Female,49,62,48
103,Male,67,62,59
104,Female,26,62,55
105,Male,49,62,56
106,Female,21,62,42
107,Male,66,63,50
108,Male,54,63,46
109,Female,68,63,43
110,Male,66,63,48
111,Female,65,63,52
112,Female,19,63,54
113,Female,38,63,42
114,Male,19,63,44
115,Female,18,63,46
116,Female,19,64,46
117,Male,63,64,51
118,Female,49,65,46
119,Female,51,65,62
120,Male,50,65,42
121,Male,27,67,59
122,Female,38,67,47
123,Female,40,69,91
124,Male,39,69,28
125,Female,23,70,29
126,Female,31,70,77
127,Male,43,71,35
128,Female,40,71,95
129,Female,59,71,11
130,Male,38,71,75
131,Male,47,71,9
132,Male,39,71,75
133,Female,25,72,34
134,Male,31,72,71
135,Female,20,73,5
136,Male,29,73,88
137,Male,44,73,7
138,Female,45,73,73
139,Male,51,74,10
140,Female,35,74,72
141,Male,27,74,5
142,Female,23,74,73
143,Female,41,75,6
144,Male,31,75,87
145,Female,20,76,6
146,Female,29,76,87
147,Male,55,77,13
148,Female,37,77,79
149,Male,46,77,16
150,Female,23,77,83
151,Male,48,78,18
152,Female,20,78,82
153,Male,59,78,20
154,Female,36,78,75
155,Male,54,78,13
156,Female,22,78,92
157,Male,34,78,15
158,Female,22,78,88
159,Male,52,79,12
160,Female,24,79,56
161,Female,47,79,14
162,Female,30,79,52
163,Female,41,81,13
164,Female,23,81,93
165,Male,41,85,18
166,Female,21,85,87
167,Male,52,86,11
168,Female,33,86,93
169,Male,60,87,13
170,Female,38,87,75
171,Male,37,87,12
172,Female,34,87,79
173,Female,53,88,16
174,Female,41,88,72
175,Female,51,88,18
176,Female,23,88,95
177,Female,34,88,14
178,Female,43,88,92
179,Female,43,93,14
180,Female,39,93,90
181,Male,54,97,15
182,Female,38,97,81
183,Male,47,98,15
184,Female,35,98,78
185,Male,45,99,13
186,Female,32,99,80
187,Male,46,101,16
188,Female,31,101,79
189,Female,54,103,17
190,Female,29,103,83
191,Male,45,103,16
192,Female,35,103,85
193,Male,34,113,14
194,Female,32,113,92
195,Male,33,120,13
196,Female,38,120,79
197,Male,47,126,15
198,Female,35,126,79
199,Male,45,137,13
200,Female,32,137,83"""


def load_data() -> pd.DataFrame:
    """Load the Mall Customers sample dataset."""
    df = pd.read_csv(StringIO(SAMPLE_CSV))
    return df


def introduce_missing(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Randomly introduce ~5 % missing values so we can demo handling.
    In a real project, the raw data already has missing values.
    Works dynamically on any numeric columns present.
    """
    rng = np.random.default_rng(seed)
    df_missing = df.copy()
    num_cols = df_missing.select_dtypes(include=[np.number]).columns.tolist()
    n_missing = max(1, int(0.05 * len(df)))
    for col in num_cols:
        idx = rng.choice(df_missing.index, size=max(1, n_missing // len(num_cols)), replace=False)
        df_missing.loc[idx, col] = np.nan
    return df_missing


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill numeric missing values with column median.
    Median is robust to outliers — better than mean for skewed data.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    return df_clean


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode ALL object/categorical columns dynamically.
    Works for any dataset, not just Gender.
    """
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include=["object", "category"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[f"{col}_Encoded"] = le.fit_transform(df_enc[col].astype(str))
        df_enc = df_enc.drop(columns=[col])
    return df_enc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering — dynamic version:
    - If AnnualIncome and SpendingScore exist → create TotalValue
    - If Age exists → create AgeGroup_Encoded
    """
    df_feat = df.copy()
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()

    # Only add TotalValue if both columns exist
    if "AnnualIncome" in num_cols and "SpendingScore" in num_cols:
        df_feat["TotalValue"] = (
            df_feat["AnnualIncome"] * df_feat["SpendingScore"] / 100
        ).round(2)

    # Only add AgeGroup if Age exists
    if "Age" in num_cols:
        bins = [0, 30, 50, 100]
        labels = [0, 1, 2]  # Young, Middle, Senior
        df_feat["AgeGroup_Encoded"] = pd.cut(
            df_feat["Age"], bins=bins, labels=labels
        ).astype(int)

    return df_feat


def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Apply StandardScaler to the selected feature columns.
    Also applies SimpleImputer to handle any remaining NaN values.
    Returns the scaled array and the fitted scaler object.
    StandardScaler → mean=0, std=1 → required for K-Means & PCA.
    """
    # First impute NaNs before scaling
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df[feature_cols])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, scaler


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric feature columns (exclude common ID columns)."""
    exclude = {"CustomerID", "ID", "Id", "index"}
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]


def apply_variance_threshold(df: pd.DataFrame, feature_cols: list, threshold: float = 0.01) -> list:
    """
    Feature Selection: remove features with variance below the threshold.
    Low-variance features contribute little to model discrimination.
    Returns the filtered list of feature column names.
    """
    if len(feature_cols) == 0:
        return feature_cols

    # Impute before computing variance
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df[feature_cols])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected_mask = selector.get_support()
    selected_cols = [col for col, selected in zip(feature_cols, selected_mask) if selected]
    return selected_cols if len(selected_cols) > 0 else feature_cols


def apply_correlation_filter(df: pd.DataFrame, feature_cols: list, corr_threshold: float = 0.95) -> list:
    """
    Feature Selection: remove highly correlated features (above corr_threshold).
    When two features are highly correlated, one is redundant.
    Returns a reduced list of feature columns.
    """
    if len(feature_cols) < 2:
        return feature_cols

    # Impute before computing correlation
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df[feature_cols])
    df_temp = pd.DataFrame(X_imputed, columns=feature_cols)

    corr_matrix = df_temp.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    kept = [col for col in feature_cols if col not in to_drop]
    return kept if len(kept) > 0 else feature_cols


def full_pipeline(df_raw: pd.DataFrame):
    """
    Run the full preprocessing pipeline and return intermediate DataFrames
    for display on the Preprocessing page.
    Works with any dataset dynamically.
    """
    df_missing = introduce_missing(df_raw)
    df_clean = handle_missing(df_missing)
    df_enc = encode_categoricals(df_clean)
    df_feat = engineer_features(df_enc)
    feature_cols = get_feature_cols(df_feat)

    # Impute before scaling (critical for NaN fix)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df_feat[feature_cols])
    df_feat_clean = df_feat.copy()
    df_feat_clean[feature_cols] = X_imputed

    X_scaled, scaler = scale_features(df_feat_clean, feature_cols)
    return {
        "raw": df_raw,
        "with_missing": df_missing,
        "cleaned": df_clean,
        "encoded": df_enc,
        "featured": df_feat,
        "featured_clean": df_feat_clean,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
