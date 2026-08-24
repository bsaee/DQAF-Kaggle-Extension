import re
import pandas as pd
from typing import Optional, Dict, Any

TARGET_NAME_PATTERNS = [
    r"^target$", r"^label$", r"^class$", r"^churn$", r"^survived$",
    r"^status$", r"^is_", r"^has_", r"^outcome$", r"^y$"
]

def is_categorical_or_discrete(series: pd.Series) -> bool:
    """Checks if a series behaves like a discrete classification target."""
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1:
        return False
    # If explicitly object, category, or bool
    if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        return n_unique <= 20
    # If integer with small distinct count
    if pd.api.types.is_integer_dtype(series):
        return n_unique <= 20
    # If float, only treat as discrete if distinct values are very low and non-fractional
    if pd.api.types.is_float_dtype(series):
        # Floats are rarely classification targets unless encoded as 0.0, 1.0
        is_integer_like = (series.dropna() % 1 == 0).all()
        return is_integer_like and (n_unique <= 10)
    return False

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    columns = list(df.columns)
    
    # 1. Match by naming conventions first
    for col in columns:
        col_lower = str(col).strip().lower()
        for pattern in TARGET_NAME_PATTERNS:
            if re.search(pattern, col_lower) and is_categorical_or_discrete(df[col]):
                return col

    # 2. Check the last column
    last_col = columns[-1]
    if is_categorical_or_discrete(df[last_col]):
        return last_col

    # 3. Find binary columns
    for col in reversed(columns):
        if df[col].nunique(dropna=True) == 2:
            return col

    # 4. Find low-cardinality discrete columns
    for col in reversed(columns):
        if is_categorical_or_discrete(df[col]):
            return col

    return None

def detect_task_type(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    selected_target = target_column if (target_column and target_column in df.columns) else infer_target_column(df)

    if selected_target is None:
        return {
            "target_column": None,
            "task_type": "unsupervised",
            "is_supported": False,
            "reason": "No discrete target detected. Dataset is structured for unsupervised learning or clustering.",
            "cardinality": None,
            "available_columns": list(df.columns)
        }

    target_series = df[selected_target].dropna()
    n_unique = target_series.nunique()

    if n_unique == 2:
        task_type = "binary_classification"
        is_supported = True
        reason = "Binary classification target identified."
    elif 3 <= n_unique <= 20:
        task_type = "multiclass_classification"
        is_supported = True
        reason = f"Multiclass classification target identified ({n_unique} classes)."
    elif pd.api.types.is_numeric_dtype(target_series) and n_unique > 20:
        task_type = "regression"
        is_supported = False
        reason = f"Continuous target detected ({n_unique} unique numeric values). Regression is outside classification stress-test scope."
    else:
        task_type = "high_cardinality_unsupported"
        is_supported = False
        reason = f"Target '{selected_target}' has {n_unique} unique values (likely an ID or text column)."

    return {
        "target_column": selected_target,
        "task_type": task_type,
        "is_supported": is_supported,
        "reason": reason,
        "cardinality": n_unique,
        "available_columns": list(df.columns)
    }