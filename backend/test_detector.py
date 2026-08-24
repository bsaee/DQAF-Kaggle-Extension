import pandas as pd
from app.engine.task_detector import detect_task_type

# 1. Test Binary Classification (e.g., Titanic-style data)
df_titanic = pd.DataFrame({
    "Age": [22.0, 38.0, 26.0, 35.0, None],
    "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
    "Survived": [0, 1, 1, 0, 0]
})
print("Titanic Auto-Detect:", detect_task_type(df_titanic))

# 2. Test User Override
print("Titanic User Override:", detect_task_type(df_titanic, target_column="Age"))

# 3. Test Unsupervised
df_unsupervised = pd.DataFrame({
    "Feature1": [1.2, 3.4, 5.6],
    "Feature2": [7.8, 9.0, 1.1]
})
print("Unsupervised Test:", detect_task_type(df_unsupervised))