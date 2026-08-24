import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def compute_pr_auc(y_true, y_probs) -> float:
    """Calculates Area Under the Precision-Recall Curve."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return float(auc(recall, precision))

def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Creates a leakage-safe preprocessing pipeline for numeric and categorical features."""
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )
    return preprocessor

def run_algorithmic_stress_test(
    df: pd.DataFrame,
    target_col: str,
    max_sample_rows: int = 5000,
    n_splits: int = 3
) -> Dict[str, Any]:
    """
    Evaluates 3 baseline architectures across stratified folds and simulates
    feature degradation impacts.
    """
    clean_df = df.dropna(subset=[target_col]).copy()
    
    # Cap and stratify sample for computational guardrail
    if len(clean_df) > max_sample_rows:
        clean_df = clean_df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(max_sample_rows / clean_df[target_col].nunique())), random_state=42)
        ).reset_index(drop=True)

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    # Convert y to numerical labels
    classes, y_encoded = np.unique(y, return_inverse=True)
    is_binary = len(classes) == 2

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=50, random_state=42)
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    baseline_metrics = {name: {"f1_scores": [], "pr_auc_scores": []} for name in models}
    
    # Run Baseline Stratified Cross-Validation
    for train_idx, val_idx in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        preprocessor = build_preprocessing_pipeline(X_train)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        for name, clf in models.items():
            clf.fit(X_train_proc, y_train)
            preds = clf.predict(X_val_proc)
            
            f1 = f1_score(y_val, preds, average="binary" if is_binary else "weighted")
            baseline_metrics[name]["f1_scores"].append(float(f1))

            if is_binary and hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val_proc)[:, 1]
                pr_auc_val = compute_pr_auc(y_val, probs)
                baseline_metrics[name]["pr_auc_scores"].append(pr_auc_val)

    # Compute Averages
    summary_scores = {}
    for name, data in baseline_metrics.items():
        avg_f1 = float(np.mean(data["f1_scores"]))
        avg_pr_auc = float(np.mean(data["pr_auc_scores"])) if is_binary else None
        summary_scores[name] = {
            "baseline_f1": round(avg_f1, 3),
            "baseline_pr_auc": round(avg_pr_auc, 3) if avg_pr_auc is not None else "N/A"
        }

    # Simulate Missingness Degradation Curve (0% to 50% induced missingness)
    degradation_curve = {"missingness_levels": [0.0, 0.1, 0.25, 0.5], "models": {}}
    for name in models:
        degradation_curve["models"][name] = []

    train_idx, val_idx = next(skf.split(X, y_encoded))
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)

    for noise_level in degradation_curve["missingness_levels"]:
        X_val_corrupted = X_val.copy()
        if noise_level > 0:
            # Mask random elements with NaN
            mask = np.random.rand(*X_val_corrupted.shape) < noise_level
            X_val_corrupted = X_val_corrupted.mask(mask)

        X_val_proc = preprocessor.transform(X_val_corrupted)

        for name, clf in models.items():
            clf.fit(X_train_proc, y_train)
            preds = clf.predict(X_val_proc)
            f1 = float(f1_score(y_val, preds, average="binary" if is_binary else "weighted"))
            degradation_curve["models"][name].append(round(f1, 3))

    # Determine Model Impact Snapshot Recommendation
    lr_drop = degradation_curve["models"]["Logistic Regression"][0] - degradation_curve["models"]["Logistic Regression"][-1]
    gb_drop = degradation_curve["models"]["Gradient Boosting"][0] - degradation_curve["models"]["Gradient Boosting"][-1]

    return {
        "baseline_summary": summary_scores,
        "degradation_curve": degradation_curve,
        "model_snapshot": {
            "linear_model_stability": "High Risk" if lr_drop > 0.15 else "Moderate Risk" if lr_drop > 0.05 else "Stable",
            "tree_model_stability": "High Risk" if gb_drop > 0.15 else "Moderate Risk" if gb_drop > 0.05 else "Stable",
            "lr_estimated_degradation_pct": round(lr_drop * 100, 1),
            "tree_estimated_degradation_pct": round(gb_drop * 100, 1)
        }
    }