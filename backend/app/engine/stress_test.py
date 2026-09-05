import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ID_PATTERNS = [r"^id$", r".*_id$", r"^id_.*", r"^uuid$", r"^guid$", r"^index$", r"^row_id$"]

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes identified unique index and key columns prior to ML benchmarking."""
    cols_to_drop = []
    for col in df.columns:
        col_clean = str(col).strip().lower()
        n_unique = df[col].nunique(dropna=True)
        n_total = len(df[col].dropna())
        if n_total > 10 and (n_unique / n_total) > 0.85:
            for p in ID_PATTERNS:
                if re.search(p, col_clean):
                    cols_to_drop.append(col)
                    break
    return df.drop(columns=cols_to_drop, errors="ignore")

def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Separates true continuous numerics from categoricals (including pseudo-numeric codes)."""
    raw_num = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

    num_cols = []
    for col in raw_num:
        clean = X[col].dropna()
        n_unique = clean.nunique()
        # Treat low-cardinality discrete integers/flags as categorical encodings
        is_discrete = (n_unique <= 2) or (pd.api.types.is_integer_dtype(X[col]) and n_unique <= 10)
        if is_discrete:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        ))

    return ColumnTransformer(transformers=transformers)

def run_classification_stress_test(df: pd.DataFrame, target_col: str, max_sample_rows: int = 5000) -> Dict[str, Any]:
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = drop_id_columns(clean_df)

    if len(clean_df) > max_sample_rows:
        clean_df = clean_df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(max_sample_rows / clean_df[target_col].nunique())), random_state=42)
        ).reset_index(drop=True)

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]
    classes, y_encoded = np.unique(y, return_inverse=True)
    is_binary = len(classes) == 2

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=50, random_state=42)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    degradation_curve = {"missingness_levels": [0.0, 0.1, 0.25, 0.5], "models": {name: [] for name in models}}

    train_idx, val_idx = next(skf.split(X, y_encoded))
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)

    for noise in degradation_curve["missingness_levels"]:
        X_val_corrupted = X_val.copy()
        if noise > 0:
            mask = np.random.rand(*X_val_corrupted.shape) < noise
            X_val_corrupted = X_val_corrupted.mask(mask)

        X_val_proc = preprocessor.transform(X_val_corrupted)

        for name, clf in models.items():
            clf.fit(X_train_proc, y_train)
            preds = clf.predict(X_val_proc)
            f1 = float(f1_score(y_val, preds, average="binary" if is_binary else "weighted"))
            degradation_curve["models"][name].append(round(f1, 3))

    lr_drop = degradation_curve["models"]["Logistic Regression"][0] - degradation_curve["models"]["Logistic Regression"][-1]
    gb_drop = degradation_curve["models"]["Gradient Boosting"][0] - degradation_curve["models"]["Gradient Boosting"][-1]

    return {
        "task_mode": "classification",
        "primary_metric": "F1-Score",
        "degradation_curve": degradation_curve,
        "model_snapshot": {
            "linear_model_stability": "High Risk" if lr_drop > 0.15 else "Moderate Risk" if lr_drop > 0.05 else "Stable",
            "tree_model_stability": "High Risk" if gb_drop > 0.15 else "Moderate Risk" if gb_drop > 0.05 else "Stable",
            "linear_drop_pct": round(lr_drop * 100, 1),
            "tree_drop_pct": round(gb_drop * 100, 1)
        }
    }

def run_regression_stress_test(df: pd.DataFrame, target_col: str, max_sample_rows: int = 5000) -> Dict[str, Any]:
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = drop_id_columns(clean_df)

    if len(clean_df) > max_sample_rows:
        clean_df = clean_df.sample(max_sample_rows, random_state=42).reset_index(drop=True)

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col].astype(float)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=50, random_state=42)
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    degradation_curve = {"missingness_levels": [0.0, 0.1, 0.25, 0.5], "models": {name: [] for name in models}}

    train_idx, val_idx = next(kf.split(X, y))
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)

    for noise in degradation_curve["missingness_levels"]:
        X_val_corrupted = X_val.copy()
        if noise > 0:
            mask = np.random.rand(*X_val_corrupted.shape) < noise
            X_val_corrupted = X_val_corrupted.mask(mask)

        X_val_proc = preprocessor.transform(X_val_corrupted)

        for name, reg in models.items():
            reg.fit(X_train_proc, y_train)
            preds = reg.predict(X_val_proc)
            r2 = float(r2_score(y_val, preds))
            degradation_curve["models"][name].append(round(max(r2, 0.0), 3))

    r_drop = degradation_curve["models"]["Ridge Regression"][0] - degradation_curve["models"]["Ridge Regression"][-1]
    gb_drop = degradation_curve["models"]["Gradient Boosting"][0] - degradation_curve["models"]["Gradient Boosting"][-1]

    return {
        "task_mode": "regression",
        "primary_metric": "R² Score",
        "degradation_curve": degradation_curve,
        "model_snapshot": {
            "linear_model_stability": "High Risk" if r_drop > 0.20 else "Moderate Risk" if r_drop > 0.08 else "Stable",
            "tree_model_stability": "High Risk" if gb_drop > 0.20 else "Moderate Risk" if gb_drop > 0.08 else "Stable",
            "linear_drop_pct": round(r_drop * 100, 1),
            "tree_drop_pct": round(gb_drop * 100, 1)
        }
    }

def run_unsupervised_profiler(df: pd.DataFrame, max_sample_rows: int = 5000) -> Dict[str, Any]:
    sample_df = drop_id_columns(df)
    if len(sample_df) > max_sample_rows:
        sample_df = sample_df.sample(max_sample_rows, random_state=42)

    preprocessor = build_preprocessing_pipeline(sample_df)
    X_proc = preprocessor.fit_transform(sample_df)

    n_components = min(X_proc.shape[1], 10)
    pca = PCA(n_components=n_components)
    pca.fit(X_proc)

    exp_variance = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
    cumulative_variance = [round(float(v) * 100, 1) for v in np.cumsum(pca.explained_variance_ratio_)]

    labels = [f"PC{i+1}" for i in range(len(exp_variance))]

    numeric_df = sample_df.select_dtypes(include=[np.number])
    variances = numeric_df.var().dropna().to_dict()
    scale_disparity = False
    if len(variances) > 1:
        vals = list(variances.values())
        if max(vals) / (min(vals) + 1e-9) > 1000:
            scale_disparity = True

    return {
        "task_mode": "unsupervised",
        "primary_metric": "Cumulative Variance %",
        "pca_scree": {
            "labels": labels,
            "individual_variance": exp_variance,
            "cumulative_variance": cumulative_variance
        },
        "unsupervised_snapshot": {
            "scale_disparity": scale_disparity,
            "components_for_80_pct": int(np.argmax(np.array(cumulative_variance) >= 80.0) + 1) if max(cumulative_variance) >= 80.0 else n_components,
            "total_effective_dimensions": X_proc.shape[1]
        }
    }

def run_task_benchmark(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    task_type: str = "classification",
    max_sample_rows: int = 5000
) -> Dict[str, Any]:
    if task_type == "unsupervised" or not target_col:
        return run_unsupervised_profiler(df, max_sample_rows)
    elif task_type == "regression":
        return run_regression_stress_test(df, target_col, max_sample_rows)
    else:
        return run_classification_stress_test(df, target_col, max_sample_rows)