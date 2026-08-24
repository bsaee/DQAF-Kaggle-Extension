import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def compute_shannon_entropy(series: pd.Series) -> float:
    """Calculates Shannon Entropy (base 2) using pure NumPy."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    probabilities = clean_series.value_counts(normalize=True).values
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))

def compute_chi2(confusion_matrix: pd.DataFrame) -> float:
    """Calculates Chi-Square statistic from contingency table using pure NumPy."""
    observed = confusion_matrix.values
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    if total == 0:
        return 0.0

    expected = (row_sums @ col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_stat = np.where(expected > 0, ((observed - expected) ** 2) / expected, 0.0)

    return float(np.sum(chi2_stat))

def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculates Cramér's V statistic without SciPy dependency."""
    confusion_matrix = pd.crosstab(x.fillna("MISSING"), y.fillna("MISSING"))
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return 0.0
    
    chi2 = compute_chi2(confusion_matrix)
    phi2 = chi2 / n
    return float(np.sqrt(phi2 / min(k - 1, r - 1)))

def calculate_dqaf_health_score(
    missing_ratio: float,
    duplicate_ratio: float,
    high_cardinality_cols: List[str],
    has_severe_imbalance: bool,
    has_moderate_imbalance: bool,
    has_leakage_suspect: bool,
    collinear_pairs: List[Dict[str, Any]]
) -> int:
    score = 100.0

    # Penalties
    score -= min(missing_ratio * 100 * 0.5, 25.0)
    score -= min(duplicate_ratio * 100 * 1.0, 10.0)
    score -= min(len(high_cardinality_cols) * 5.0, 15.0)
    
    if has_severe_imbalance:
        score -= 15.0
    elif has_moderate_imbalance:
        score -= 8.0

    if has_leakage_suspect:
        score -= 20.0

    score -= min(len(collinear_pairs) * 3.0, 15.0)

    return int(max(round(score), 0))

def profile_dataset(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    # 1. Basic Vitals
    missing_cells = int(df.isnull().sum().sum())
    missing_ratio = float(missing_cells / total_cells) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ratio = float(duplicate_rows / total_rows) if total_rows > 0 else 0.0

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # 2. Per-Column Metrics & Shannon Entropy
    column_profiles = []
    high_cardinality_cols = []
    for col in df.columns:
        col_series = df[col]
        n_unique = col_series.nunique(dropna=True)
        null_count = int(col_series.isnull().sum())
        null_pct = float(null_count / total_rows * 100) if total_rows > 0 else 0.0
        
        ent = compute_shannon_entropy(col_series)
        
        is_high_cardinality = False
        if col in categorical_cols and n_unique > 50 and n_unique > (0.2 * total_rows):
            is_high_cardinality = True
            high_cardinality_cols.append(col)

        column_profiles.append({
            "name": col,
            "dtype": str(col_series.dtype),
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
            "unique_count": n_unique,
            "shannon_entropy": round(ent, 3),
            "is_high_cardinality": is_high_cardinality
        })

    # 3. Class Imbalance Check
    has_severe_imbalance = False
    has_moderate_imbalance = False
    class_distribution = {}
    if target_column and target_column in df.columns:
        target_counts = df[target_column].value_counts(normalize=True)
        class_distribution = {str(k): round(float(v) * 100, 2) for k, v in target_counts.items()}
        if len(target_counts) > 1:
            maj_ratio = target_counts.iloc[0]
            if maj_ratio >= 0.80:
                has_severe_imbalance = True
            elif maj_ratio >= 0.65:
                has_moderate_imbalance = True

    # 4. Multicollinearity Matrix
    collinear_pairs = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(method="pearson").abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and val >= 0.85:
                    collinear_pairs.append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": round(float(val), 3)
                    })

    # 5. Categorical Association Matrix (Cramér's V)
    cramers_matrix = {}
    if len(categorical_cols) > 1:
        for c1 in categorical_cols:
            cramers_matrix[c1] = {}
            for c2 in categorical_cols:
                if c1 == c2:
                    cramers_matrix[c1][c2] = 1.0
                else:
                    cramers_matrix[c1][c2] = round(compute_cramers_v(df[c1], df[c2]), 3)

    # 6. Target Leakage Suspect Check
    has_leakage_suspect = False
    leakage_suspects = []
    if target_column and target_column in df.columns:
        for col in df.columns:
            if col == target_column:
                continue
            if col in numeric_cols and target_column in numeric_cols:
                corr = abs(df[col].corr(df[target_column]))
                if not np.isnan(corr) and corr >= 0.98:
                    has_leakage_suspect = True
                    leakage_suspects.append(col)
            elif col in categorical_cols:
                v = compute_cramers_v(df[col], df[target_column])
                if v >= 0.98:
                    has_leakage_suspect = True
                    leakage_suspects.append(col)

    # 7. DQAF Health Score
    health_score = calculate_dqaf_health_score(
        missing_ratio=missing_ratio,
        duplicate_ratio=duplicate_ratio,
        high_cardinality_cols=high_cardinality_cols,
        has_severe_imbalance=has_severe_imbalance,
        has_moderate_imbalance=has_moderate_imbalance,
        has_leakage_suspect=has_leakage_suspect,
        collinear_pairs=collinear_pairs
    )

    return {
        "vitals": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "missing_cells": missing_cells,
            "missing_cell_pct": round(missing_ratio * 100, 2),
            "duplicate_rows": duplicate_rows,
            "duplicate_row_pct": round(duplicate_ratio * 100, 2),
            "numeric_column_count": len(numeric_cols),
            "categorical_column_count": len(categorical_cols)
        },
        "health_score": health_score,
        "risk_flags": {
            "has_severe_imbalance": has_severe_imbalance,
            "has_moderate_imbalance": has_moderate_imbalance,
            "has_leakage_suspect": has_leakage_suspect,
            "leakage_columns": leakage_suspects,
            "high_cardinality_columns": high_cardinality_cols,
            "collinear_pairs": collinear_pairs
        },
        "class_distribution": class_distribution,
        "column_profiles": column_profiles,
        "cramers_matrix": cramers_matrix
    }