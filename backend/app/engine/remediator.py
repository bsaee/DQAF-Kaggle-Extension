from typing import Dict, Any, List

def generate_remediation_recipes(
    risk_flags: Dict[str, Any],
    target_col: Optional[str] = None,
    task_mode: str = "classification",
    regression_stats: Optional[Dict[str, float]] = None
) -> List[Dict[str, str]]:
    recipes = []

    # ------------------ CLASSIFICATION RECIPES ------------------
    if task_mode == "classification" and target_col:
        if risk_flags.get("has_severe_imbalance") or risk_flags.get("has_moderate_imbalance"):
            severity = "Severe" if risk_flags.get("has_severe_imbalance") else "Moderate"
            recipes.append({
                "title": f"Balanced Class Weighting ({severity} Imbalance)",
                "description": f"Target '{target_col}' shows {severity.lower()} class imbalance. Use class weighting to prevent biased decision boundaries.",
                "code": (
                    "from sklearn.utils.class_weight import compute_class_weight\n"
                    "import numpy as np\n\n"
                    f"# Compute inverse class frequencies for '{target_col}'\n"
                    "classes = np.unique(y_train)\n"
                    "weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)\n"
                    "class_weight_dict = dict(zip(classes, weights))\n\n"
                    "# Pass to model:\n"
                    "# LogisticRegression(class_weight='balanced')\n"
                    "# RandomForestClassifier(class_weight='balanced')"
                )
            })

    # ------------------ REGRESSION RECIPES ------------------
    if task_mode == "regression" and target_col:
        if risk_flags.get("has_high_skew"):
            skew_val = regression_stats.get("skewness", 0.0) if regression_stats else "High"
            recipes.append({
                "title": f"Power / Log Transformation (Target Skew = {skew_val})",
                "description": f"Continuous target '{target_col}' is heavily skewed. Stabilize variance and residuals using PowerTransformer or log1p.",
                "code": (
                    "import numpy as np\n"
                    "from sklearn.preprocessing import PowerTransformer\n\n"
                    f"# Apply Yeo-Johnson power transform to target '{target_col}'\n"
                    "pt = PowerTransformer(method='yeo-johnson')\n"
                    "y_train_trans = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()\n"
                    "y_test_trans = pt.transform(y_test.values.reshape(-1, 1)).flatten()\n\n"
                    "# Invert predictions back to original scale:\n"
                    "# y_pred_original = pt.inverse_transform(y_pred.reshape(-1, 1))"
                )
            })

        recipes.append({
            "title": "Regularized Regression Pipeline (Ridge/Lasso)",
            "description": "Standardize features and penalize large coefficients to prevent overfitting on collinear continuous features.",
            "code": (
                "from sklearn.pipeline import make_pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import RidgeCV\n\n"
                "ridge_pipeline = make_pipeline(\n"
                "    StandardScaler(),\n"
                "    RidgeCV(alphas=[0.1, 1.0, 10.0])\n"
                ")\n"
                "ridge_pipeline.fit(X_train, y_train)"
            )
        })

    # ------------------ UNSUPERVISED / CLUSTERING RECIPES ------------------
    if task_mode == "unsupervised":
        recipes.append({
            "title": "Standard Scaling + PCA Dimensionality Reduction",
            "description": "Distance-based algorithms (K-Means, DBSCAN, PCA) are sensitive to unscaled magnitudes. Standardize features before projection.",
            "code": (
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.decomposition import PCA\n"
                "from sklearn.pipeline import make_pipeline\n\n"
                "# Retain 95% of cumulative explained variance\n"
                "pca_pipeline = make_pipeline(\n"
                "    StandardScaler(),\n"
                "    PCA(n_components=0.95)\n"
                ")\n"
                "X_projected = pca_pipeline.fit_transform(X)"
            )
        })

    # ------------------ GENERAL DATA CLEANING RECIPES ------------------
    collinear_pairs = risk_flags.get("collinear_pairs", [])
    if collinear_pairs:
        cols_to_drop = list({pair["col2"] for pair in collinear_pairs})
        recipes.append({
            "title": "Drop Highly Collinear Features",
            "description": f"Identified {len(collinear_pairs)} collinear pair(s) (r >= 0.85). Drop redundant features to stabilize models.",
            "code": (
                f"# Redundant collinear features to remove:\n"
                f"collinear_cols_to_drop = {cols_to_drop}\n\n"
                "X_train_reduced = X_train.drop(columns=collinear_cols_to_drop)\n"
                "X_test_reduced = X_test.drop(columns=collinear_cols_to_drop)"
            )
        })

    high_card_cols = risk_flags.get("high_cardinality_columns", [])
    if high_card_cols and task_mode != "unsupervised":
        recipes.append({
            "title": "Target Encoding for High Cardinality",
            "description": f"Features {high_card_cols} have high cardinality. Use TargetEncoder to prevent dimension explosion.",
            "code": (
                "from sklearn.preprocessing import TargetEncoder\n\n"
                f"high_card_cols = {high_card_cols}\n"
                "encoder = TargetEncoder(smooth='auto', cv=5)\n"
                "X_train[high_card_cols] = encoder.fit_transform(X_train[high_card_cols], y_train)\n"
                "X_test[high_card_cols] = encoder.transform(X_test[high_card_cols])"
            )
        })

    if not recipes:
        recipes.append({
            "title": "Standard Imputation & Scaling Pipeline",
            "description": "No critical structural flaws detected. Standard median imputation and scaling pipeline recommended.",
            "code": (
                "from sklearn.pipeline import make_pipeline\n"
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.preprocessing import StandardScaler\n\n"
                "pipeline = make_pipeline(\n"
                "    SimpleImputer(strategy='median'),\n"
                "    StandardScaler()\n"
                ")\n"
                "X_train_proc = pipeline.fit_transform(X_train)\n"
                "X_test_proc = pipeline.transform(X_test)"
            )
        })

    return recipes