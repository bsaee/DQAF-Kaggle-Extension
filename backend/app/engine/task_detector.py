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
    if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        return n_unique <= 20
    if pd.api.types.is_integer_dtype(series):
        return n_unique <= 20
    if pd.api.types.is_float_dtype(series):
        is_integer_like = (series.dropna() % 1 == 0).all()
        return is_integer_like and (n_unique <= 10)
    return False

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    columns = list(df.columns)
    for col in columns:
        col_lower = str(col).strip().lower()
        for pattern in TARGET_NAME_PATTERNS:
            if re.search(pattern, col_lower) and is_categorical_or_discrete(df[col]):
                return col

    last_col = columns[-1]
    if is_categorical_or_discrete(df[last_col]):
        return last_col

    for col in reversed(columns):
        if df[col].nunique(dropna=True) == 2:
            return col

    for col in reversed(columns):
        if is_categorical_or_discrete(df[col]):
            return col

    return None

def detect_task_type(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_mode: str = "auto"
) -> Dict[str, Any]:
    # 1. Unsupervised / Clustering Mode
    if task_mode == "unsupervised":
        return {
            "target_column": None,
            "task_type": "unsupervised",
            "is_supported": False,
            "reason": "Unsupervised / Clustering mode selected by user.",
            "cardinality": None,
            "available_columns": list(df.columns)
        }

    # 2. Resolve Target Column
    selected_target = target_column if (target_column and target_column in df.columns) else infer_target_column(df)

    # 3. Regression Mode
    if task_mode == "regression":
        n_unique = df[selected_target].nunique() if selected_target else 0
        return {
            "target_column": selected_target,
            "task_type": "regression",
            "is_supported": False,
            "reason": "Regression task selected. Classification stress-tests are disabled.",
            "cardinality": n_unique,
            "available_columns": list(df.columns)
        }

    # 4. Explicit Classification Mode
    if task_mode == "classification":
        if selected_target is None:
            selected_target = df.columns[-1]
        n_unique = df[selected_target].nunique()
        return {
            "target_column": selected_target,
            "task_type": "binary_classification" if n_unique == 2 else "multiclass_classification",
            "is_supported": True,
            "reason": f"Classification mode enforced on '{selected_target}'.",
            "cardinality": n_unique,
            "available_columns": list(df.columns)
        }

    # 5. Auto-Detection (Default)
    if selected_target is None:
        return {
            "target_column": None,
            "task_type": "unsupervised",
            "is_supported": False,
            "reason": "No discrete target detected. Auto-routed to Unsupervised/Clustering.",
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
        reason = f"Continuous target detected ({n_unique} unique values). Regression task."
    else:
        task_type = "high_cardinality_unsupported"
        is_supported = False
        reason = f"Target '{selected_target}' has {n_unique} unique values."

    return {
        "target_column": selected_target,
        "task_type": task_type,
        "is_supported": is_supported,
        "reason": reason,
        "cardinality": n_unique,
        "available_columns": list(df.columns)
    }