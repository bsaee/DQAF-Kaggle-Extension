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
from sklearn.metrics import f1_score, precision_recall_curve, auc, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

def run_classification_stress_test(df: pd.DataFrame, target_col: str, max_sample_rows: int = 5000) -> Dict[str, Any]:
    clean_df = df.dropna(subset=[target_col]).copy()
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
            # Bound R2 display at 0 minimum for clean plotting
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
    sample_df = df.sample(min(len(df), max_sample_rows), random_state=42) if len(df) > max_sample_rows else df.copy()

    preprocessor = build_preprocessing_pipeline(sample_df)
    X_proc = preprocessor.fit_transform(sample_df)

    n_components = min(X_proc.shape[1], 10)
    pca = PCA(n_components=n_components)
    pca.fit(X_proc)

    exp_variance = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
    cumulative_variance = [round(float(v) * 100, 1) for v in np.cumsum(pca.explained_variance_ratio_)]

    labels = [f"PC{i+1}" for i in range(len(exp_variance))]

    # Features scale disparities
    numeric_df = df.select_dtypes(include=[np.number])
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
    """Routes to the proper benchmark engine based on active task mode."""
    if task_type == "unsupervised" or not target_col:
        return run_unsupervised_profiler(df, max_sample_rows)
    elif task_type == "regression":
        return run_regression_stress_test(df, target_col, max_sample_rows)
    else:
        return run_classification_stress_test(df, target_col, max_sample_rows)