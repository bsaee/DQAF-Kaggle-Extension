from app.engine.remediator import generate_remediation_recipes

sample_risk_flags = {
    "has_severe_imbalance": True,
    "has_leakage_suspect": False,
    "leakage_columns": [],
    "high_cardinality_columns": ["City_Code"],
    "collinear_pairs": [{"col1": "FeatureA", "col2": "FeatureB", "correlation": 0.95}]
}

recipes = generate_remediation_recipes(
    risk_flags=sample_risk_flags,
    target_col="Churn",
    class_distribution={"0": 92.0, "1": 8.0}
)

for r in recipes:
    print(f"--- {r['title']} ---")
    print(r['description'])
    print(r['code'])
    print()