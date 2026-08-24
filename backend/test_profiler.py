import pandas as pd
from app.engine.profiler import profile_dataset

# Create a sample dataset with deliberate flaws:
# - Collinear features (FeatureA & FeatureB)
# - Imbalanced target (90% class 0)
# - Missing values in Age
df_sample = pd.DataFrame({
    "Age": [22.0, None, 26.0, 35.0, 54.0, None, 29.0, 42.0, 31.0, 19.0],
    "FeatureA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "FeatureB": [1.01, 1.99, 3.02, 3.98, 5.01, 6.02, 6.99, 8.01, 9.02, 10.0], # Collinear with A
    "Category": ["A", "B", "A", "A", "B", "A", "B", "A", "A", "B"],
    "Target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # Severe imbalance
})

profile = profile_dataset(df_sample, target_column="Target")

print(f"DQAF Health Score: {profile['health_score']}/100")
print("Risk Flags:", profile["risk_flags"])
print("Vitals:", profile["vitals"])