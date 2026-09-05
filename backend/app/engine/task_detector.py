import re
import pandas as pd
from typing import Optional, Dict, Any

TARGET_NAME_PATTERNS = [
    r"^target$", r"^label$", r"^class$", r"^churn$", r"^survived$",
    r"^status$", r"^is_", r"^has_", r"^outcome$", r"^y$"
]

ID_PATTERNS = [
    r"^id$", r".*_id$", r"^id_.*", r"^uuid$", r"^guid$", r"^index$", r"^row_id$"
]

def is_id_col(col: str, series: pd.Series) -> bool:
    col_lower = str(col).strip().lower()
    n_unique = series.nunique(dropna=True)
    n_total = len(series.dropna())
    if n_total > 10 and (n_unique / n_total) > 0.85:
        for p in ID_PATTERNS:
            if re.search(p, col_lower):
                return True
    return False

def is_categorical_or_discrete(series: pd.Series) -> bool:
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
    # Candidates excluding obvious IDs
    candidate_cols = [c for c in df.columns if not is_id_col(c, df[c])]

    for col in candidate_cols:
        col_lower = str(col).strip().lower()
        for pattern in TARGET_NAME_PATTERNS:
            if re.search(pattern, col_lower) and is_categorical_or_discrete(df[col]):
                return col

    for col in reversed(candidate_cols):
        if df[col].nunique(dropna=True) == 2:
            return col

    for col in reversed(candidate_cols):
        if is_categorical_or_discrete(df[col]):
            return col

    return candidate_cols[-1] if candidate_cols else None

def detect_task_type(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_mode: str = "auto"
) -> Dict[str, Any]:
    # Filter target options to exclude ID columns
    usable_columns = [c for c in df.columns if not is_id_col(c, df[c])]

    if task_mode == "unsupervised":
        return {
            "target_column": None,
            "task_type": "unsupervised",
            "is_supported": False,
            "reason": "Unsupervised mode selected.",
            "cardinality": None,
            "available_columns": usable_columns
        }

    selected_target = target_column if (target_column and target_column in df.columns) else infer_target_column(df)

    if task_mode == "regression":
        n_unique = df[selected_target].nunique() if selected_target else 0
        return {
            "target_column": selected_target,
            "task_type": "regression",
            "is_supported": False,
            "reason": "Regression task selected.",
            "cardinality": n_unique,
            "available_columns": usable_columns
        }

    if task_mode == "classification":
        if selected_target is None and usable_columns:
            selected_target = usable_columns[-1]
        n_unique = df[selected_target].nunique() if selected_target else 0
        return {
            "target_column": selected_target,
            "task_type": "binary_classification" if n_unique == 2 else "multiclass_classification",
            "is_supported": True,
            "reason": f"Classification enforced on '{selected_target}'.",
            "cardinality": n_unique,
            "available_columns": usable_columns
        }

    if selected_target is None:
        return {
            "target_column": None,
            "task_type": "unsupervised",
            "is_supported": False,
            "reason": "No valid target identified. Defaulted to unsupervised.",
            "cardinality": None,
            "available_columns": usable_columns
        }

    target_series = df[selected_target].dropna()
    n_unique = target_series.nunique()

    if n_unique == 2:
        task_type = "binary_classification"
        is_supported = True
        reason = "Binary classification target detected."
    elif 3 <= n_unique <= 20:
        task_type = "multiclass_classification"
        is_supported = True
        reason = f"Multiclass target detected ({n_unique} classes)."
    elif pd.api.types.is_numeric_dtype(target_series) and n_unique > 20:
        task_type = "regression"
        is_supported = False
        reason = "Continuous regression target detected."
    else:
        task_type = "high_cardinality_unsupported"
        is_supported = False
        reason = f"High cardinality target ({n_unique} distinct values)."

    return {
        "target_column": selected_target,
        "task_type": task_type,
        "is_supported": is_supported,
        "reason": reason,
        "cardinality": n_unique,
        "available_columns": usable_columns
    }