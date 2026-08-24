import pandas as pd
import numpy as np
from app.engine.stress_test import run_algorithmic_stress_test

# Create synthetic dataset with non-linear relations and categorical features
np.random.seed(42)
n_rows = 200
df_synthetic = pd.DataFrame({
    "Numeric1": np.random.randn(n_rows),
    "Numeric2": np.random.randn(n_rows) * 2,
    "Category": np.random.choice(["X", "Y", "Z"], size=n_rows),
    "Outcome": np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
})

results = run_algorithmic_stress_test(df_synthetic, target_col="Outcome")
print("Baseline Summary:", results["baseline_summary"])
print("Model Snapshot:", results["model_snapshot"])
print("Degradation Curve:", results["degradation_curve"])