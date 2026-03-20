import os
import joblib  # Thư viện dùng để lưu model
import json
import time
import warnings
import importlib
import subprocess
import sys
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =====================
# Install packages if missing
# =====================
def ensure_package(package_name, import_name=None):
    module_name = import_name if import_name is not None else package_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

ensure_package("xgboost")
ensure_package("lightgbm")
ensure_package("tabulate")

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 200)

# =====================
# Config
# =====================
CSV_PATH = "data/stage2_final.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
SCORING = "neg_root_mean_squared_error"

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/metadata", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# =====================
# Helpers
# =====================
def read_csv_robust(path):
    encodings = ["utf-8", "utf-8-sig", "cp1258", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_total_candidates(param_distributions):
    return len(ParameterGrid(param_distributions))

def print_markdown_table(df_, title=None):
    if title:
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
    try:
        print(df_.to_markdown(index=False))
    except Exception:
        print(df_)

def make_one_hot_encoder(drop_first):
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

# =====================
# Custom Transformers
# =====================
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

# =====================
# Preprocessors
# =====================
def build_preprocessor(model_family):
    if model_family == "ols":
        drop_first = True
    elif model_family in ["regularized", "tree"]:
        drop_first = False
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    return ColumnTransformer(
        transformers=[
            ("log_area", LogAreaTransformer(), ["Area_m2"]),
            ("furniture", FurnitureEncoder(), ["Furniture"]),
            ("district", make_one_hot_encoder(drop_first=drop_first), ["District"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def build_pipeline(model, model_family, use_scaler):
    steps = [("preprocessor", build_preprocessor(model_family))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps=steps)

def evaluate_model(y_test_log, y_pred_log):
    y_test_raw = np.expm1(y_test_log)
    y_pred_raw = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    r2 = r2_score(y_test_raw, y_pred_raw)
    mape = safe_mape(y_test_raw, y_pred_raw)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }

def get_feature_names_from_pipeline(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    names = list(preprocessor.get_feature_names_out())
    cleaned = []
    for n in names:
        if n.startswith("district_"):
            raw = n.replace("district_", "")
            cleaned.append(district_to_feature_token(raw))
        else:
            cleaned.append(n)
    return cleaned

def get_reference_district(fitted_pipeline):
    district_encoder = fitted_pipeline.named_steps["preprocessor"].named_transformers_["district"]
    return district_encoder.categories_[0][0]

def build_ols_effects_table(ols_pipeline):
    core_model = ols_pipeline.named_steps["model"]
    feature_names = get_feature_names_from_pipeline(ols_pipeline)

    intercept = float(core_model.intercept_)
    coef = np.ravel(core_model.coef_)

    df_ = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef
    })
    df_["Abs_Coefficient"] = df_["Coefficient"].abs()

    def effect_text(row):
        f = row["Feature"]
        b = row["Coefficient"]
        if f == "Log_Area_m2":
            return f"Elasticity ≈ {b:.4f}"
        return f"{(np.exp(b) - 1) * 100:.2f}%"

    df_["Effect"] = df_.apply(effect_text, axis=1)
    df_ = df_.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

    return intercept, df_[["Feature", "Coefficient", "Effect"]].head(10)

# =====================
# Load Data
# =====================
df = read_csv_robust(CSV_PATH)
df.columns = [str(c).strip() for c in df.columns]
df["District"] = df["District"].astype(str).map(normalize_district_title)
df["Furniture"] = df["Furniture"].astype(str).str.strip()

required_cols = [
    "Title",
    "Price_Million_VND",
    "Area_m2",
    "Price_Per_M2",
    "Bedroom",
    "Toilet",
    "Furniture",
    "District",
    "Link",
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

X_raw = df[["Area_m2", "Furniture", "District"]].copy()
y_raw = pd.to_numeric(df["Price_Per_M2"], errors="coerce")
y_log = np.log1p(y_raw)

if y_log.isna().any():
    raise ValueError("Price_Per_M2 contains NaN/non-numeric values after coercion.")

X_train_raw, X_test_raw, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X_raw, y_log, y_raw,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# =====================
# Model Specs
# =====================
model_specs = [
    {
        "name": "Multiple Linear Regression",
        "model": LinearRegression(),
        "model_family": "ols",
        "use_scaler": False,
        "param_distributions": {"model__fit_intercept": [True]},
        "n_iter_cap": 1,
        "search": False,
    },
    {
        "name": "Ridge Regression",
        "model": Ridge(),
        "model_family": "regularized",
        "use_scaler": True,
        "param_distributions": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        "n_iter_cap": 5,
        "search": True,
    },
    {
        "name": "Lasso Regression",
        "model": Lasso(max_iter=20000, random_state=RANDOM_STATE),
        "model_family": "regularized",
        "use_scaler": True,
        "param_distributions": {"model__alpha": [0.0001, 0.0005, 0.001, 0.01]},
        "n_iter_cap": 4,
        "search": True,
    },
    {
        "name": "Elastic Net Regression",
        "model": ElasticNet(max_iter=20000, random_state=RANDOM_STATE),
        "model_family": "regularized",
        "use_scaler": True,
        "param_distributions": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        },
        "n_iter_cap": 8,
        "search": True,
    },
    {
        "name": "K-Nearest Neighbors Regression",
        "model": KNeighborsRegressor(),
        "model_family": "regularized",
        "use_scaler": True,
        "param_distributions": {
            "model__n_neighbors": [5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        },
        "n_iter_cap": 8,
        "search": True,
    },
    {
        "name": "Support Vector Regression",
        "model": SVR(kernel="rbf"),
        "model_family": "regularized",
        "use_scaler": True,
        "param_distributions": {
            "model__C": [0.5, 1, 5, 10],
            "model__epsilon": [0.05, 0.1, 0.2],
            "model__gamma": ["scale"]
        },
        "n_iter_cap": 6,
        "search": True,
    },
    {
        "name": "Random Forest Regression",
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "model_family": "tree",
        "use_scaler": False,
        "param_distributions": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        "n_iter_cap": 8,
        "search": True,
    },
    {
        "name": "Gradient Boosting Regression",
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "model_family": "tree",
        "use_scaler": False,
        "param_distributions": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0]
        },
        "n_iter_cap": 8,
        "search": True,
    },
    {
        "name": "XGBoost Regression",
        "model": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
            eval_metric="rmse"
        ),
        "model_family": "tree",
        "use_scaler": False,
        "param_distributions": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_lambda": [1, 5]
        },
        "n_iter_cap": 8,
        "search": True,
    },
    {
        "name": "LightGBM Regression",
        "model": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        "model_family": "tree",
        "use_scaler": False,
        "param_distributions": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31],
            "model__max_depth": [-1, 5, 10],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        },
        "n_iter_cap": 8,
        "search": True,
    },
]

# =====================
# Train + Tune + Evaluate
# =====================
all_results = []
fitted_estimators = {}

for spec in model_specs:
    model_name = spec["name"]
    print(f"\n{'='*100}\nRunning: {model_name}\n{'='*100}")
    start_time = time.time()

    try:
        pipeline = build_pipeline(
            model=spec["model"],
            model_family=spec["model_family"],
            use_scaler=spec["use_scaler"]
        )

        if spec["search"] is False:
            pipeline.fit(X_train_raw, y_train_log)
            best_estimator = pipeline
            best_params = {}
            cv_best_rmse_log = np.nan
        else:
            total_candidates = get_total_candidates(spec["param_distributions"])
            n_iter = min(spec["n_iter_cap"], total_candidates)

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=spec["param_distributions"],
                n_iter=n_iter,
                scoring=SCORING,
                cv=CV_FOLDS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
                verbose=0,
                error_score="raise"
            )

            search.fit(X_train_raw, y_train_log)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            cv_best_rmse_log = -search.best_score_

        y_pred_log = best_estimator.predict(X_test_raw)
        metrics = evaluate_model(y_test_log, y_pred_log)

        elapsed = time.time() - start_time
        fitted_estimators[model_name] = best_estimator

        all_results.append({
            "Model": model_name,
            "Status": "Success",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "MAPE": metrics["MAPE"],
            "CV_Best_RMSE_LogScale": cv_best_rmse_log,
            "Best_Params": best_params,
            "Runtime_Seconds": elapsed,
            "Error": None
        })

        print(f"[SUCCESS] {model_name} | RMSE={metrics['RMSE']:.4f} | Time={elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        all_results.append({
            "Model": model_name,
            "Status": "Failed",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "CV_Best_RMSE_LogScale": np.nan,
            "Best_Params": None,
            "Runtime_Seconds": elapsed,
            "Error": f"{type(e).__name__}: {str(e)}"
        })
        print(f"[FAILED] {model_name}: {type(e).__name__}: {e}")

results_df = pd.DataFrame(all_results)
results_df["sort_key"] = np.where(results_df["Status"] == "Success", 0, 1)
results_df = results_df.sort_values(
    by=["sort_key", "RMSE"],
    ascending=[True, True],
    na_position="last"
).drop(columns=["sort_key"]).reset_index(drop=True)

results_display_df = results_df.copy()
for col in ["MAE", "RMSE", "MAPE", "CV_Best_RMSE_LogScale", "Runtime_Seconds"]:
    results_display_df[col] = results_display_df[col].round(2)
results_display_df["R2"] = results_display_df["R2"].round(4)

print_markdown_table(
    results_display_df[["Model", "Status", "MAE", "RMSE", "R2", "MAPE", "Runtime_Seconds"]],
    title="MODEL COMPARISON"
)

# =====================
# OLS Table
# =====================
ols_intercept = None
ols_effects_df = None

if "Multiple Linear Regression" in fitted_estimators:
    ols_pipeline = fitted_estimators["Multiple Linear Regression"]
    ols_intercept, ols_effects_df = build_ols_effects_table(ols_pipeline)

# =====================
# Summary
# =====================
success_df = results_df[results_df["Status"] == "Success"].copy()
if len(success_df) == 0:
    raise RuntimeError("No model finished successfully.")

best_model_name = success_df.sort_values("RMSE", ascending=True).iloc[0]["Model"]
best_pipeline = fitted_estimators[best_model_name]

reference_district = get_reference_district(best_pipeline)
final_feature_names = get_feature_names_from_pipeline(best_pipeline)

summary_info = {
    "n_rows": int(df.shape[0]),
    "n_original_columns": int(df.shape[1]),
    "n_final_features_after_encoding": len(final_feature_names),
    "final_features": final_feature_names,
    "reference_district_dropped_by_OHE": reference_district,
    "best_model": best_model_name
}

print("\n" + "=" * 100)
print("SUMMARY INFO")
print("=" * 100)
print(json.dumps(summary_info, ensure_ascii=False, indent=2))

print("\nBEST MODEL:", best_model_name)
print("REFERENCE DISTRICT:", reference_district)

if ols_effects_df is not None:
    print(f"\nOLS INTERCEPT: {ols_intercept:.6f}")
    print_markdown_table(ols_effects_df, title="OLS TOP COEFFICIENTS FOR PAPER")

print("\n" + "=" * 100)
print("BEST PARAMS BY MODEL")
print("=" * 100)
for _, row in results_df.iterrows():
    print(f"\n{row['Model']}")
    print("Status:", row["Status"])
    print("Best_Params:", row["Best_Params"])

# =====================
# Save outputs
# =====================
results_df.to_csv("outputs/tables/model_comparison_results.csv", index=False, encoding="utf-8-sig")

if ols_effects_df is not None:
    ols_effects_df.to_csv("outputs/tables/ols_top_coefficients_for_paper.csv", index=False, encoding="utf-8-sig")

with open("outputs/metadata/experiment_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_info, f, ensure_ascii=False, indent=2)


# ==========================================
# LƯU MODEL (.PKL)
# ==========================================
model_save_path = "outputs/models/best_model.pkl"
joblib.dump(best_pipeline, model_save_path)
print(f"\n>>> [THÀNH CÔNG] Đã lưu toàn bộ Pipeline mô hình ({best_model_name}) tại: {model_save_path}")
