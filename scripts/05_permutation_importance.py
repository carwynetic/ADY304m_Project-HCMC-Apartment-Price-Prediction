import os
import re
import json
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

CSV_PATH = "data/stage2_final.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs("outputs/tables", exist_ok=True)

# =========================================================
# Helpers
# =========================================================
def read_csv_robust(path):
    encodings = ["utf-8", "utf-8-sig", "cp1258", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error

def normalize_text_basic(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_district_title(text):
    return normalize_text_basic(text).title()

def remove_accents(text):
    text = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def district_to_feature_token(text):
    text = normalize_district_title(text)
    text = remove_accents(text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    return f"District_{text}"

def get_total_candidates(param_distributions):
    return len(ParameterGrid(param_distributions))

def make_one_hot_encoder(drop_first=False):
    try:
        return OneHotEncoder(
            drop="first" if drop_first else None,
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            drop="first" if drop_first else None,
            handle_unknown="ignore",
            sparse=False
        )

# =========================================================
# Custom transformers
# =========================================================
class LogAreaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        else:
            s = pd.Series(np.ravel(X))
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().any():
            raise ValueError("Area_m2 contains NaN/non-numeric values.")
        if (s <= -1).any():
            raise ValueError("Area_m2 contains values <= -1, cannot apply log1p.")
        return np.log1p(s.to_numpy(dtype=float)).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Log_Area_m2"], dtype=object)

class FurnitureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        else:
            s = pd.Series(np.ravel(X))

        s = s.astype(str).str.strip().str.lower()
        mapped = s.map({
            "không": 0,
            "khong": 0,
            "ko": 0,
            "có": 1,
            "co": 1
        })

        if mapped.isna().any():
            bad_values = sorted(s[mapped.isna()].unique().tolist())
            raise ValueError(f"Unknown Furniture values found: {bad_values}")

        return mapped.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Furniture_Encoded"], dtype=object)

# =========================================================
# Load data
# =========================================================
df = read_csv_robust(CSV_PATH)
df.columns = [str(c).strip() for c in df.columns]
df["District"] = df["District"].astype(str).map(normalize_district_title)
df["Furniture"] = df["Furniture"].astype(str).str.strip()

X_raw = df[["Area_m2", "Furniture", "District"]].copy()
y_orig = pd.to_numeric(df["Price_Per_M2"], errors="coerce")
y_log = np.log1p(y_orig)

X_train_raw, X_test_raw, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X_raw, y_log, y_orig,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# =========================================================
# Build tree-based matrix (drop_first=False)
# =========================================================
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("log_area", LogAreaTransformer(), ["Area_m2"]),
        ("furniture", FurnitureEncoder(), ["Furniture"]),
        ("district", make_one_hot_encoder(drop_first=False), ["District"]),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

X_train_tree = preprocessor_tree.fit_transform(X_train_raw)
X_test_tree = preprocessor_tree.transform(X_test_raw)

feature_names_raw = list(preprocessor_tree.get_feature_names_out())
feature_names = []
for n in feature_names_raw:
    if n.startswith("district_"):
        feature_names.append(district_to_feature_token(n.replace("district_", "")))
    else:
        feature_names.append(n)

X_train_tree = pd.DataFrame(X_train_tree, columns=feature_names, index=X_train_raw.index)
X_test_tree = pd.DataFrame(X_test_tree, columns=feature_names, index=X_test_raw.index)

# =========================================================
# Train Random Forest
# =========================================================
rf_param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_distributions,
    n_iter=min(8, get_total_candidates(rf_param_distributions)),
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
    verbose=0,
    error_score="raise"
)

search.fit(X_train_tree, y_train_log)
rf_model = search.best_estimator_

# =========================================================
# Permutation Importance with RMSE on original scale
# =========================================================
def neg_rmse_on_original_scale(estimator, X, y_true):
    y_pred_log = estimator.predict(X)
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return -rmse

perm = permutation_importance(
    estimator=rf_model,
    X=X_test_tree,
    y=y_test_orig,
    scoring=neg_rmse_on_original_scale,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": X_test_tree.columns,
    "Importance_Mean": perm.importances_mean,
    "Importance_STD": perm.importances_std
})

perm_df["Importance_For_Share"] = perm_df["Importance_Mean"].clip(lower=0)

total_importance_all = perm_df["Importance_For_Share"].sum()

perm_df["Importance_Percent"] = np.where(
    total_importance_all > 0,
    perm_df["Importance_For_Share"] / total_importance_all * 100,
    0
)

top10 = perm_df.sort_values("Importance_Mean", ascending=False).head(10).copy()
top10["Importance_Mean"] = top10["Importance_Mean"].round(4)
top10["Importance_STD"] = top10["Importance_STD"].round(4)
top10["Importance_Percent"] = top10["Importance_Percent"].round(2)

print(top10[["Feature", "Importance_Mean", "Importance_STD", "Importance_Percent"]])

top10.to_csv(
    "outputs/tables/random_forest_permutation_importance_top10.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLaTeX rows:")
for _, row in top10.iterrows():
    print(
        f"\\texttt{{{row['Feature']}}} & "
        f"{row['Importance_Mean']:.4f} $\\pm$ {row['Importance_STD']:.4f} & "
        f"{row['Importance_Percent']:.2f} \\\\"
    )