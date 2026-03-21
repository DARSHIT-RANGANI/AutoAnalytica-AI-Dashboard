"""
Tests for app.services.eda_engine module.

Covers:
  1. Summary statistics (numeric, categorical, datetime)
  2. Correlation matrix
  3. Distribution data (histograms + value counts)
  4. Edge cases
"""

import numpy as np
import pandas as pd
import pytest

from app.services.eda_engine import run_eda


# ── helpers ──────────────────────────────────────────

def _make_sample_df() -> pd.DataFrame:
    """Build a small mixed-type DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "age":     [25, 30, 35, 40, 45, 50, 55, 60],
        "salary":  [3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5],
        "dept":    ["HR", "IT", "IT", "HR", "Finance", "IT", "HR", "Finance"],
        "joined":  pd.to_datetime([
            "2020-01-01", "2020-06-15", "2021-03-10", "2021-09-20",
            "2022-01-05", "2022-07-11", "2023-02-28", "2023-11-30",
        ]),
    })


# ── 1. Summary statistics ──────────────────────────

def test_summary_stats_numeric():
    df = _make_sample_df()
    report = run_eda(df)

    assert "age" in report["summary_stats"]
    assert "salary" in report["summary_stats"]

    age_stats = report["summary_stats"]["age"]
    assert "mean" in age_stats
    assert "count" in age_stats
    assert age_stats["count"] == 8.0
    assert age_stats["missing"] == 0


def test_summary_stats_categorical():
    df = _make_sample_df()
    report = run_eda(df)

    assert "dept" in report["summary_stats"]
    dept_stats = report["summary_stats"]["dept"]
    assert "unique" in dept_stats
    assert dept_stats["missing"] == 0


def test_summary_stats_datetime():
    df = _make_sample_df()
    report = run_eda(df)

    assert "joined" in report["summary_stats"]
    joined_stats = report["summary_stats"]["joined"]
    assert "min" in joined_stats
    assert "max" in joined_stats


# ── 2. Correlation matrix ──────────────────────────

def test_correlation_matrix_structure():
    df = _make_sample_df()
    report = run_eda(df)
    corr = report["correlation"]

    assert "age" in corr
    assert "salary" in corr
    # self-correlation = 1.0
    assert corr["age"]["age"] == 1.0


def test_correlation_single_numeric_col():
    df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    report = run_eda(df)
    assert report["correlation"] == {}


# ── 3. Distributions ───────────────────────────────

def test_numeric_distribution():
    df = _make_sample_df()
    report = run_eda(df)
    dist = report["distributions"]

    assert "age" in dist
    assert dist["age"]["type"] == "numeric"
    assert len(dist["age"]["bins"]) > 0
    assert len(dist["age"]["counts"]) > 0
    assert "mean" in dist["age"]
    assert "median" in dist["age"]


def test_categorical_distribution():
    df = _make_sample_df()
    report = run_eda(df)
    dist = report["distributions"]

    assert "dept" in dist
    assert dist["dept"]["type"] == "categorical"
    assert len(dist["dept"]["labels"]) <= 10
    assert dist["dept"]["unique_count"] == 3


# ── 4. Edge cases ──────────────────────────────────

def test_empty_dataframe():
    df = pd.DataFrame()
    report = run_eda(df)
    assert report["shape"] == {"rows": 0, "columns": 0}
    assert report["summary_stats"] == {}


def test_invalid_input_raises():
    with pytest.raises(TypeError):
        run_eda([1, 2, 3])


def test_shape_reported():
    df = _make_sample_df()
    report = run_eda(df)
    assert report["shape"]["rows"] == 8
    assert report["shape"]["columns"] == 4


def test_columns_listed():
    df = _make_sample_df()
    report = run_eda(df)
    col_names = [c["name"] for c in report["columns"]]
    assert "age" in col_names
    assert "salary" in col_names
    assert "dept" in col_names
    assert "joined" in col_names


def test_toggle_sections_off():
    df = _make_sample_df()
    report = run_eda(
        df,
        include_summary=False,
        include_correlation=False,
        include_distributions=False,
    )
    assert "summary_stats" not in report
    assert "correlation" not in report
    assert "distributions" not in report
    assert "shape" in report


def test_json_serialisable():
    """Ensure the entire report can survive a JSON round-trip."""
    import json
    df = _make_sample_df()
    report = run_eda(df)
    # Should not raise
    json_str = json.dumps(report, default=str)
    assert len(json_str) > 0
