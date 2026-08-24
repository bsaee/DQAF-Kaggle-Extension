from typing import Dict, Any, List

def generate_remediation_recipes(
    risk_flags: Dict[str, Any],
    target_col: str,
    class_distribution: Dict[str, float]
) -> List[Dict[str, str]]:
    """
    Generates actionable, copy-pasteable Python snippets based on detected data risks.
    """
    recipes = []

    # 1. Class Imbalance Recipe
    if risk_flags.get("has_severe_imbalance"):
        recipes.append({
            "title": "Balanced Class Weighting",
            "description": f"Target '{target_col}' has severe class imbalance. Use balanced class weights during estimator initialization.",
            "code": (
                "from sklearn.utils.class_weight import compute_class_weight\n"
                "import numpy as np\n\n"
                f"# Compute inverse class frequencies for target '{target_col}'\n"
                "classes = np.unique(y_train)\n"
                "weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)\n"
                "class_weight_dict = dict(zip(classes, weights))\n\n"
                "# Pass to model:\n"
                "# LogisticRegression(class_weight='balanced')\n"
                "# RandomForestClassifier(class_weight='balanced')\n"
                "# For LightGBM: LGBMClassifier(scale_pos_weight=weights[1]/weights[0])"
            )
        })

    # 2. Collinear Feature Drop Recipe
    collinear_pairs = risk_flags.get("collinear_pairs", [])
    if collinear_pairs:
        cols_to_drop = list({pair["col2"] for pair in collinear_pairs})
        recipes.append({
            "title": "Drop Highly Collinear Features",
            "description": f"Identified {len(collinear_pairs)} feature pairs with Pearson correlation >= 0.85. Drop redundant columns to stabilize linear models.",
            "code": (
                f"# Redundant collinear features to remove:\n"
                f"collinear_cols_to_drop = {cols_to_drop}\n\n"
                "X_train_reduced = X_train.drop(columns=collinear_cols_to_drop)\n"
                "X_test_reduced = X_test.drop(columns=collinear_cols_to_drop)"
            )
        })

    # 3. High Cardinality Encoding Recipe
    high_card_cols = risk_flags.get("high_cardinality_columns", [])
    if high_card_cols:
        recipes.append({
            "title": "Target / Frequency Encoding for High Cardinality",
            "description": f"Columns {high_card_cols} have high cardinality. One-hot encoding will explode feature dimensions; use TargetEncoder instead.",
            "code": (
                "from sklearn.preprocessing import TargetEncoder\n\n"
                f"high_card_cols = {high_card_cols}\n"
                "encoder = TargetEncoder(smooth='auto', cv=5)\n"
                "X_train[high_card_cols] = encoder.fit_transform(X_train[high_card_cols], y_train)\n"
                "X_test[high_card_cols] = encoder.transform(X_test[high_card_cols])"
            )
        })

    # 4. Target Leakage Warning / Prune Recipe
    leakage_cols = risk_flags.get("leakage_columns", [])
    if leakage_cols:
        recipes.append({
            "title": "Remove Suspected Leaky Features",
            "description": f"Features {leakage_cols} have >= 0.98 correlation/association with the target. Remove them to prevent synthetic overfitting.",
            "code": (
                f"leaky_features = {leakage_cols}\n"
                "X_clean = X.drop(columns=leaky_features)"
            )
        })

    # 5. Fallback Standard Baseline Pipeline
    if not recipes:
        recipes.append({
            "title": "Clean Data Pipeline",
            "description": "No severe structural anomalies detected. Standard median imputation and scaling pipeline recommended.",
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