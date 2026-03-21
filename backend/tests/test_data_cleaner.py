"""
Tests for app.services.data_cleaner module.

Covers all 6 cleaning features:
  1. Column name standardisation
  2. Numeric missing → median
  3. Categorical missing → mode
  4. Duplicate removal
  5. IQR outlier clipping
  6. Data-type correction
"""

import numpy as np
import pandas as pd
import pytest

from app.services.data_cleaner import clean_dataframe


# ── helpers ──────────────────────────────────────────

def _make_dirty_df() -> pd.DataFrame:
    """Build a small DataFrame with known issues."""
    return pd.DataFrame({
        "First Name":  ["Alice", "Bob", "Bob", "Charlie", None, "Eve"],
        " Age ":       [25, np.nan, 30, 30, 35, 200],        # messy name, missing, outlier
        "salary(USD)": [50_000, 60_000, 60_000, np.nan, 80_000, 90_000],
        "joined":      ["2020-01-01", "2021-06-15", "2021-06-15",
                        "2022-03-10", "2023-07-20", "2024-12-01"],
        "score":       ["10", "20", "20", "30", "40", "50"],  # numeric stored as str
    })


# ── 1. Column standardisation ──────────────────────

def test_column_names_standardised():
    df = _make_dirty_df()
    result = clean_dataframe(df)
    cleaned = result["cleaned_df"]

    assert "first_name" in cleaned.columns
    assert "age" in cleaned.columns
    assert "salary_usd" in cleaned.columns
    assert "First Name" not in cleaned.columns
    assert " Age " not in cleaned.columns


def test_columns_renamed_summary():
    df = _make_dirty_df()
    result = clean_dataframe(df)
    renamed = result["summary"]["columns_renamed"]

    assert "First Name" in renamed
    assert " Age " in renamed
    assert "salary(USD)" in renamed


# ── 2. Numeric missing → median ────────────────────

def test_numeric_missing_filled_with_median():
    df = _make_dirty_df()
    result = clean_dataframe(df, clip_outliers=False, correct_dtypes=False)
    cleaned = result["cleaned_df"]

    # " Age " → "age"; the original NaN should be replaced by the median of
    # [25, 30, 30, 35, 200] = 30
    assert cleaned["age"].isnull().sum() == 0
    assert result["summary"]["missing_filled"]["numeric"] >= 1


# ── 3. Categorical missing → mode ──────────────────

def test_categorical_missing_filled_with_mode():
    df = _make_dirty_df()
    result = clean_dataframe(df, clip_outliers=False, correct_dtypes=False)
    cleaned = result["cleaned_df"]

    assert cleaned["first_name"].isnull().sum() == 0
    assert result["summary"]["missing_filled"]["categorical"] >= 1


# ── 4. Duplicate removal ───────────────────────────

def test_duplicates_removed():
    df = _make_dirty_df()
    result = clean_dataframe(df, clip_outliers=False, correct_dtypes=False)
    cleaned = result["cleaned_df"]

    assert result["summary"]["duplicates_removed"] >= 1
    assert len(cleaned) < len(df)


# ── 5. IQR outlier clipping ────────────────────────

def test_outliers_clipped():
    df = pd.DataFrame({"value": [10, 12, 14, 13, 11, 15, 100]})
    result = clean_dataframe(df)

    assert result["summary"]["outliers_clipped"] >= 1
    cleaned = result["cleaned_df"]
    assert cleaned["value"].max() < 100  # 100 should have been clipped


# ── 6. Data-type correction ────────────────────────

def test_dtype_correction_numeric():
    df = pd.DataFrame({"nums": ["1", "2", "3", "4"]})
    result = clean_dataframe(df)
    cleaned = result["cleaned_df"]

    assert pd.api.types.is_numeric_dtype(cleaned["nums"])
    assert "nums" in result["summary"]["types_corrected"]


def test_dtype_correction_datetime():
    df = pd.DataFrame({"dates": ["2020-01-01", "2021-06-15", "2022-03-10"]})
    result = clean_dataframe(df)
    cleaned = result["cleaned_df"]

    assert pd.api.types.is_datetime64_any_dtype(cleaned["dates"])
    assert "dates" in result["summary"]["types_corrected"]


# ── Edge cases ──────────────────────────────────────

def test_empty_dataframe():
    df = pd.DataFrame()
    result = clean_dataframe(df)
    assert result["cleaned_df"].empty
    assert result["summary"]["duplicates_removed"] == 0


def test_invalid_input_raises():
    with pytest.raises(TypeError):
        clean_dataframe("not a dataframe")


def test_toggle_steps_off():
    """All cleaning steps disabled → output equals input."""
    df = _make_dirty_df()
    result = clean_dataframe(
        df,
        standardise_columns=False,
        fill_numeric=False,
        fill_categorical=False,
        remove_duplicates=False,
        clip_outliers=False,
        correct_dtypes=False,
    )
    cleaned = result["cleaned_df"]
    pd.testing.assert_frame_equal(cleaned, df)
