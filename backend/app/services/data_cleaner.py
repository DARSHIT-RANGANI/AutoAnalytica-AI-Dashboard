"""
data_cleaner.py — Reusable data-cleaning pipeline for AutoAnalytica AI.

Features
--------
1. Standardise column names
2. Fill missing numeric values with median
3. Fill missing categorical values with mode
4. Remove duplicate rows
5. Auto data-type correction  ← runs BEFORE outlier clipping (FIX-2)
6. IQR-based outlier detection & clipping

Fixes applied vs original
--------------------------
FIX-1  _standardise_columns: duplicate-name collision now produces unique
        suffixed names (e.g. first_name, first_name_2) instead of creating
        two columns with identical names that break downstream alignment.

FIX-2  clean_dataframe step order: correct_dtypes now runs BEFORE
        clip_outliers so that numeric strings ("50000") are converted to
        real numbers before outlier bounds are computed.

FIX-3  _clip_outliers_iqr: added np.isnan(iqr) guard so all-NaN columns
        are skipped explicitly instead of relying on NaN-comparison
        behaviour, which is fragile across pandas versions.

FIX-4  _correct_dtypes: format="mixed" is only available in pandas >= 2.0.
        On older installations the call now falls back gracefully to
        infer_datetime_format=True (pandas 1.x) instead of raising a
        TypeError that would silently swallow the entire step.

Usage
-----
    from app.services.data_cleaner import clean_dataframe

    result = clean_dataframe(df)
    cleaned_df       = result["cleaned_df"]
    cleaning_summary = result["summary"]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Detect pandas version once at import time (used in FIX-4)
_PANDAS_MAJOR, _PANDAS_MINOR = (
    int(x) for x in pd.__version__.split(".")[:2]
)
_PANDAS_GE_2 = _PANDAS_MAJOR >= 2


# ───────────────────────────────────────────────
# 1. Standardise column names
# ───────────────────────────────────────────────

def _standardise_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Lowercase, strip whitespace, and replace spaces / special characters
    with underscores.

    FIX-1: If two source columns normalise to the same target name,
    append a numeric suffix (_2, _3 …) so every output column is unique.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with renamed columns.
    rename_map : dict
        Mapping of {original_name: new_name} for columns that changed.
    """
    seen: Dict[str, str] = {}          # original_col → resolved clean name
    resolved_names: set = set()        # all clean names committed so far

    for col in df.columns:
        new_name = col.strip().lower()
        new_name = re.sub(r"[^a-z0-9_]", "_", new_name)   # non-alnum → _
        new_name = re.sub(r"_+", "_", new_name)            # collapse runs
        new_name = new_name.strip("_") or "col"            # never empty

        # FIX-1: collision resolution — append suffix until name is unique
        base_name  = new_name
        suffix_idx = 2
        while new_name in resolved_names:
            new_name = f"{base_name}_{suffix_idx}"
            suffix_idx += 1

        seen[col] = new_name
        resolved_names.add(new_name)

    rename_map = {orig: clean for orig, clean in seen.items() if orig != clean}
    if rename_map:
        df = df.rename(columns=rename_map)

    return df, rename_map


# ───────────────────────────────────────────────
# 2. Fill missing numeric values with median
# ───────────────────────────────────────────────

def _fill_numeric_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Fill NaN values in numeric columns with the column median.

    Columns whose median is itself NaN (e.g. all-NaN columns) are
    left untouched — there is no sensible fill value.
    """
    total_filled = 0
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        missing_count = int(df[col].isnull().sum())
        if missing_count == 0:
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            logger.debug("Skipping '%s' — median is NaN (all-NaN column)", col)
            continue
        df[col] = df[col].fillna(median_val)
        total_filled += missing_count
        logger.debug(
            "Filled %d missing values in '%s' with median %.4f",
            missing_count, col, median_val,
        )

    return df, total_filled


# ───────────────────────────────────────────────
# 3. Fill missing categorical values with mode
# ───────────────────────────────────────────────

def _fill_categorical_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Fill NaN values in object / category / string columns with the mode.

    All-NaN columns (mode() returns an empty Series) are skipped.
    """
    total_filled = 0
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns

    for col in cat_cols:
        missing_count = int(df[col].isnull().sum())
        if missing_count == 0:
            continue
        mode_values = df[col].mode()
        if mode_values.empty:
            logger.debug("Skipping '%s' — mode is empty (all-NaN column)", col)
            continue
        fill_val = mode_values.iloc[0]
        df[col] = df[col].fillna(fill_val)
        total_filled += missing_count
        logger.debug(
            "Filled %d missing values in '%s' with mode '%s'",
            missing_count, col, fill_val,
        )

    return df, total_filled


# ───────────────────────────────────────────────
# 4. Remove duplicate rows
# ───────────────────────────────────────────────

def _remove_duplicates(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows and return the count removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.debug("Removed %d duplicate rows", removed)
    return df, removed


# ───────────────────────────────────────────────
# 5. Data-type correction  (FIX-2: runs before outlier clipping)
# ───────────────────────────────────────────────

def _correct_dtypes(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Attempt to convert object columns to more appropriate types:
      • numeric  (via pd.to_numeric)
      • datetime (via pd.to_datetime)

    Running this BEFORE clip_outliers (FIX-2) means numeric-looking
    strings ("50000") become real numbers and are subject to IQR clipping.

    FIX-4: pd.to_datetime with format="mixed" is only available in
    pandas >= 2.0.  On earlier versions we fall back to
    infer_datetime_format=True so the step degrades gracefully instead
    of raising a TypeError that would be swallowed by the caller.
    """
    corrections: Dict[str, str] = {}
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        # Try numeric first
        try:
            converted = pd.to_numeric(df[col], errors="raise")
            df[col] = converted
            corrections[col] = str(converted.dtype)
            logger.debug("Converted '%s' to %s", col, converted.dtype)
            continue
        except (ValueError, TypeError):
            pass

        # Try datetime — FIX-4: format="mixed" only on pandas >= 2.0
        try:
            if _PANDAS_GE_2:
                converted = pd.to_datetime(
                    df[col], errors="raise", format="mixed"
                )
            else:
                # pandas 1.x fallback — no format="mixed" parameter
                converted = pd.to_datetime(
                    df[col], errors="raise", infer_datetime_format=True
                )
            df[col] = converted
            corrections[col] = str(converted.dtype)
            logger.debug("Converted '%s' to datetime", col)
        except (ValueError, TypeError, OverflowError):
            pass

    return df, corrections


# ───────────────────────────────────────────────
# 6. IQR outlier detection & clipping
# ───────────────────────────────────────────────

def _clip_outliers_iqr(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, int]:
    """
    For each numeric column, clip values that fall outside
    [Q1 − factor×IQR,  Q3 + factor×IQR].

    FIX-3: Columns where IQR is 0 OR NaN (e.g. constant or all-NaN
    columns) are skipped explicitly using pd.isna() / np.isnan() instead
    of relying on NaN-comparison returning False, which is fragile across
    pandas versions.
    """
    total_clipped = 0
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        # FIX-3: skip constant columns AND all-NaN columns
        if iqr == 0 or pd.isna(iqr):
            continue

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        outlier_mask  = (df[col] < lower) | (df[col] > upper)
        clipped_count = int(outlier_mask.sum())

        if clipped_count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_clipped += clipped_count
            logger.debug(
                "Clipped %d outliers in '%s' (bounds %.4f – %.4f)",
                clipped_count, col, lower, upper,
            )

    return df, total_clipped


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def clean_dataframe(
    df: pd.DataFrame,
    *,
    standardise_columns: bool = True,
    fill_numeric:        bool = True,
    fill_categorical:    bool = True,
    remove_duplicates:   bool = True,
    correct_dtypes:      bool = True,
    clip_outliers:       bool = True,
    iqr_factor:          float = 1.5,
) -> Dict[str, Any]:
    """
    Run the full cleaning pipeline on *df* and return a result dict.

    Return value
    ------------
    {
        "cleaned_df": pd.DataFrame,
        "summary": {
            "columns_renamed":    {old: new, ...},
            "missing_filled":     {"numeric": int, "categorical": int},
            "duplicates_removed": int,
            "types_corrected":    {col: dtype_str, ...},
            "outliers_clipped":   int,
        }
    }

    Step order
    ----------
    1. Standardise column names  (FIX-1: collision-safe deduplication)
    2. Fill missing numeric      (median; skips all-NaN columns)
    3. Fill missing categorical  (mode;   skips all-NaN columns)
    4. Remove duplicate rows
    5. Correct data types        (FIX-2: before outlier clipping;
                                  FIX-4: pandas-version-safe datetime)
    6. Clip outliers (IQR)       (FIX-3: NaN-safe IQR guard)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.  A copy is taken internally — the original
        is never mutated.
    standardise_columns : bool
        Lowercase & sanitise column names with collision deduplication.
    fill_numeric : bool
        Fill numeric NaNs with column median.
    fill_categorical : bool
        Fill categorical NaNs with column mode.
    remove_duplicates : bool
        Drop exact duplicate rows.
    correct_dtypes : bool
        Auto-convert object columns to numeric / datetime.
    clip_outliers : bool
        Clip values outside iqr_factor × IQR.
    iqr_factor : float
        Multiplier for IQR outlier bounds (default 1.5).

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if df.empty:
        logger.warning("Received an empty DataFrame — nothing to clean.")
        return {
            "cleaned_df": df.copy(),
            "summary": {
                "columns_renamed":    {},
                "missing_filled":     {"numeric": 0, "categorical": 0},
                "duplicates_removed": 0,
                "types_corrected":    {},
                "outliers_clipped":   0,
            },
        }

    # Work on a copy — caller's DataFrame is never mutated
    df = df.copy()
    summary: Dict[str, Any] = {}

    # ── 1. Column names ────────────────────────────────────────────────────────
    columns_renamed: Dict[str, str] = {}
    if standardise_columns:
        try:
            df, columns_renamed = _standardise_columns(df)
        except Exception:
            logger.exception("Error standardising column names")
    summary["columns_renamed"] = columns_renamed

    # ── 2 & 3. Missing values ──────────────────────────────────────────────────
    numeric_filled = categorical_filled = 0

    if fill_numeric:
        try:
            df, numeric_filled = _fill_numeric_missing(df)
        except Exception:
            logger.exception("Error filling numeric missing values")

    if fill_categorical:
        try:
            df, categorical_filled = _fill_categorical_missing(df)
        except Exception:
            logger.exception("Error filling categorical missing values")

    summary["missing_filled"] = {
        "numeric":     numeric_filled,
        "categorical": categorical_filled,
    }

    # ── 4. Duplicates ──────────────────────────────────────────────────────────
    duplicates_removed = 0
    if remove_duplicates:
        try:
            df, duplicates_removed = _remove_duplicates(df)
        except Exception:
            logger.exception("Error removing duplicates")
    summary["duplicates_removed"] = duplicates_removed

    # ── 5. Type correction  (FIX-2: before outlier clipping) ──────────────────
    types_corrected: Dict[str, str] = {}
    if correct_dtypes:
        try:
            df, types_corrected = _correct_dtypes(df)
        except Exception:
            logger.exception("Error correcting data types")
    summary["types_corrected"] = types_corrected

    # ── 6. Outliers  (FIX-3 NaN-safe guard inside _clip_outliers_iqr) ─────────
    outliers_clipped = 0
    if clip_outliers:
        try:
            df, outliers_clipped = _clip_outliers_iqr(df, factor=iqr_factor)
        except Exception:
            logger.exception("Error clipping outliers")
    summary["outliers_clipped"] = outliers_clipped

    logger.info(
        "Cleaning complete — renamed %d cols, filled %d numeric / %d categorical, "
        "removed %d duplicates, corrected %d types, clipped %d outliers",
        len(columns_renamed),
        numeric_filled,
        categorical_filled,
        duplicates_removed,
        len(types_corrected),
        outliers_clipped,
    )

    return {"cleaned_df": df, "summary": summary}


# ═══════════════════════════════════════════════
# Self-tests  (python data_cleaner.py)
# ═══════════════════════════════════════════════

def _run_tests() -> None:
    print("\n── data_cleaner.py self-tests ──")

    # ── 1. FIX-1: Duplicate column name collision ──────────────────────────────
    df = pd.DataFrame(
        [[1, 2, 3]],
        columns=["First Name", "first name", "FIRST NAME"],
    )
    result = clean_dataframe(df)
    cols = list(result["cleaned_df"].columns)
    assert len(set(cols)) == 3, f"Expected 3 unique cols, got {cols}"
    assert cols[0] == "first_name"
    assert cols[1] == "first_name_2"
    assert cols[2] == "first_name_3"
    renamed = result["summary"]["columns_renamed"]
    assert len(renamed) == 3
    print(f"✓ FIX-1 duplicate column names resolved: {cols}")

    # ── 2. FIX-2: correct_dtypes runs before clip_outliers ────────────────────
    # Column "salary" contains numeric strings — should be converted AND clipped
    df = pd.DataFrame({
        "salary": ["10000", "20000", "30000", "40000", "1000000"],  # 1M is outlier
    })
    result = clean_dataframe(df)
    cleaned = result["cleaned_df"]
    assert pd.api.types.is_numeric_dtype(cleaned["salary"]), \
        "salary should be numeric after correct_dtypes"
    assert result["summary"]["types_corrected"].get("salary") is not None
    assert result["summary"]["outliers_clipped"] > 0, \
        "1M outlier should be clipped AFTER type conversion"
    print(f"✓ FIX-2 correct_dtypes before clip_outliers  "
          f"(clipped={result['summary']['outliers_clipped']})")

    # ── 3. FIX-3: all-NaN column skipped by IQR guard ─────────────────────────
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [np.nan, np.nan, np.nan],    # all-NaN
    })
    result = clean_dataframe(df, fill_numeric=False)   # don't fill so b stays NaN
    assert result["cleaned_df"]["b"].isnull().all(), \
        "all-NaN column should remain all-NaN"
    print("✓ FIX-3 all-NaN column skipped by IQR guard")

    # ── 4. Missing numeric fill with median ────────────────────────────────────
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
    result = clean_dataframe(df, clip_outliers=False)
    assert result["cleaned_df"]["x"].isnull().sum() == 0
    assert result["summary"]["missing_filled"]["numeric"] == 2
    # median of [1, 3, 5] = 3.0
    filled_vals = result["cleaned_df"]["x"].tolist()
    assert 3.0 in filled_vals, f"Expected 3.0 fill, got {filled_vals}"
    print(f"✓ Numeric missing fill (median=3.0): {filled_vals}")

    # ── 5. Missing categorical fill with mode ──────────────────────────────────
    # Use unique values so deduplication after fill does not shrink the DataFrame
    df = pd.DataFrame({"cat": ["a", "b", "c", None, "d"]})
    result = clean_dataframe(df, clip_outliers=False, correct_dtypes=False,
                             remove_duplicates=False)
    assert result["cleaned_df"]["cat"].isnull().sum() == 0
    assert result["summary"]["missing_filled"]["categorical"] == 1
    filled_val = result["cleaned_df"]["cat"].iloc[3]
    assert filled_val is not None
    print(f"✓ Categorical missing fill (mode='{filled_val}')")

    # ── 6. Duplicate removal ───────────────────────────────────────────────────
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})
    result = clean_dataframe(df, clip_outliers=False)
    assert result["summary"]["duplicates_removed"] == 1
    assert len(result["cleaned_df"]) == 3
    print("✓ Duplicate removal (1 duplicate removed)")

    # ── 7. IQR outlier clipping ────────────────────────────────────────────────
    vals = list(range(1, 21)) + [1000]     # 1000 is a clear outlier
    df = pd.DataFrame({"v": vals})
    result = clean_dataframe(df, correct_dtypes=False)
    assert result["cleaned_df"]["v"].max() < 1000, \
        "Outlier 1000 should be clipped"
    assert result["summary"]["outliers_clipped"] >= 1
    print(f"✓ IQR outlier clipping  "
          f"(clipped={result['summary']['outliers_clipped']}, "
          f"max after={result['cleaned_df']['v'].max():.1f})")

    # ── 8. Constant column skipped by IQR guard ────────────────────────────────
    df = pd.DataFrame({"c": [5, 5, 5, 5, 5], "v": [1, 2, 100, 3, 4]})
    result = clean_dataframe(df)
    assert result["cleaned_df"]["c"].nunique() == 1, \
        "Constant column 'c' must not be modified"
    print("✓ Constant column skipped by IQR guard (IQR=0)")

    # ── 9. Datetime conversion (type correction) ───────────────────────────────
    df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15", "2023-12-31"]})
    result = clean_dataframe(df, clip_outliers=False)
    assert pd.api.types.is_datetime64_any_dtype(result["cleaned_df"]["date"]), \
        "Date strings should be converted to datetime"
    assert "date" in result["summary"]["types_corrected"]
    print("✓ Datetime conversion via correct_dtypes")

    # ── 10. Empty DataFrame ────────────────────────────────────────────────────
    df_empty = pd.DataFrame()
    result = clean_dataframe(df_empty)
    assert result["cleaned_df"].empty
    assert result["summary"]["outliers_clipped"] == 0
    print("✓ Empty DataFrame handled gracefully")

    # ── 11. TypeError on non-DataFrame input ───────────────────────────────────
    try:
        clean_dataframe([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "pd.DataFrame" in str(e) or "DataFrame" in str(e)
    print("✓ TypeError raised for non-DataFrame input")

    # ── 12. Original DataFrame is not mutated (copy safety) ───────────────────
    df_orig = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    df_orig_copy = df_orig.copy()
    _ = clean_dataframe(df_orig)
    pd.testing.assert_frame_equal(df_orig, df_orig_copy)
    print("✓ Original DataFrame not mutated (copy-safe)")

    # ── 13. All steps disabled — returns unchanged copy ───────────────────────
    df = pd.DataFrame({"A Col": [1, 2, 3]})
    result = clean_dataframe(
        df,
        standardise_columns=False, fill_numeric=False,
        fill_categorical=False,    remove_duplicates=False,
        correct_dtypes=False,      clip_outliers=False,
    )
    assert list(result["cleaned_df"].columns) == ["A Col"]
    assert result["summary"]["columns_renamed"] == {}
    print("✓ All steps disabled — DataFrame passes through unchanged")

    # ── 14. All-NaN categorical column (mode-skip guard) ─────────────────────
    df = pd.DataFrame({
        "normal": ["a", "b", "a"],
        "all_nan": pd.Series([None, None, None], dtype=object),
    })
    result = clean_dataframe(df, clip_outliers=False, correct_dtypes=False)
    assert result["cleaned_df"]["all_nan"].isnull().all(), \
        "all-NaN categorical column should remain untouched"
    print("✓ All-NaN categorical column skipped (mode.empty guard)")

    print("\n✓ All data_cleaner.py self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()