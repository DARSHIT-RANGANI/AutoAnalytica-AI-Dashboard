"""
eda_engine.py — Exploratory Data Analysis engine for AutoAnalytica AI.

Generates JSON-ready analytics from a pandas DataFrame:
  1. Summary statistics (count, mean, std, min, max, quartiles …)
  2. Correlation matrix
  3. Distribution data (numeric histograms + categorical value counts)

Fixes applied vs original
--------------------------
FIX-1  Removed unused import ``List`` from ``typing``.

FIX-2  _correlation_matrix / run_eda: an invalid correlation_method now
        raises a clear ValueError immediately — before any computation —
        instead of silently returning {} with no feedback to the caller.
        The validation is performed once in run_eda (upfront) and once
        inside _correlation_matrix (for direct callers of the helper).

Usage
-----
    from app.services.eda_engine import run_eda

    report = run_eda(df)

    # With options:
    report = run_eda(
        df,
        correlation_method="spearman",
        histogram_bins=30,
        top_n_categories=5,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional   # FIX-1: 'List' removed (was unused)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FIX-2: canonical set of valid method names — used for validation in both
# run_eda() (upfront) and _correlation_matrix() (direct callers).
_VALID_CORR_METHODS = frozenset({"pearson", "spearman", "kendall"})


# ═══════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════

def _summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generate summary statistics for every column.

    Numeric columns  → count, mean, std, min, 25%, 50%, 75%, max, missing
    Categorical cols → count, unique, top, freq, missing
    Datetime cols    → count, min, max, missing
    """
    summary: Dict[str, Dict[str, Any]] = {}

    # ── Numeric ────────────────────────────────
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        stats = numeric_df.describe()
        for col in stats.columns:
            col_stats: Dict[str, Any] = {}
            for stat_name, value in stats[col].items():
                col_stats[str(stat_name)] = (
                    None if pd.isna(value) else float(value)
                )
            col_stats["missing"] = int(df[col].isnull().sum())
            col_stats["dtype"]   = str(df[col].dtype)
            summary[str(col)]    = col_stats

    # ── Categorical ────────────────────────────
    cat_df = df.select_dtypes(include=["object", "category", "string"])
    if not cat_df.empty:
        stats = cat_df.describe()
        for col in stats.columns:
            col_stats = {}
            for stat_name, value in stats[col].items():
                col_stats[str(stat_name)] = (
                    None if pd.isna(value) else
                    int(value) if isinstance(value, (int, np.integer)) else
                    str(value)
                )
            col_stats["missing"] = int(df[col].isnull().sum())
            col_stats["dtype"]   = str(df[col].dtype)
            summary[str(col)]    = col_stats

    # ── Datetime ───────────────────────────────
    dt_df = df.select_dtypes(include=["datetime", "datetimetz"])
    if not dt_df.empty:
        for col in dt_df.columns:
            series = df[col]
            min_val = series.min()
            max_val = series.max()
            col_stats = {
                "count":   int(series.count()),
                "min":     str(min_val) if not pd.isna(min_val) else None,
                "max":     str(max_val) if not pd.isna(max_val) else None,
                "missing": int(series.isnull().sum()),
                "dtype":   str(series.dtype),
            }
            summary[str(col)] = col_stats

    return summary


def _correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute a pairwise correlation matrix for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        One of ``'pearson'``, ``'spearman'``, ``'kendall'``.
        FIX-2: raises ValueError immediately for any other value.

    Returns
    -------
    dict
        Nested dict ``{col_a: {col_b: float | None, …}, …}``
        ready for JSON serialisation.  Returns ``{}`` when fewer than
        two numeric columns are present.

    Raises
    ------
    ValueError
        If *method* is not one of the three valid options.
    """
    # FIX-2: guard here so direct callers also get a clear error
    if method not in _VALID_CORR_METHODS:
        raise ValueError(
            f"correlation_method must be one of {set(_VALID_CORR_METHODS)}, "
            f"got '{method}'"
        )

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return {}

    corr = numeric_df.corr(method=method)

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for col in corr.columns:
        result[str(col)] = {
            str(idx): None if pd.isna(val) else round(float(val), 4)
            for idx, val in corr[col].items()
        }

    return result


def _distribution_data(
    df: pd.DataFrame,
    *,
    num_bins: int = 20,
    top_n_categories: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate distribution data for each column.

    Numeric columns
        → histogram with bin edges and counts, plus mean / median / std.
    Categorical columns
        → top-N value counts and total unique count.
    """
    distributions: Dict[str, Dict[str, Any]] = {}

    # ── Numeric histograms ─────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            distributions[str(col)] = {
                "type":   "numeric",
                "bins":   [],
                "counts": [],
            }
            continue

        counts, bin_edges = np.histogram(series, bins=num_bins)
        distributions[str(col)] = {
            "type":   "numeric",
            "bins":   [round(float(e), 4) for e in bin_edges],
            "counts": [int(c) for c in counts],
            "mean":   round(float(series.mean()),   4),
            "median": round(float(series.median()), 4),
            "std":    round(float(series.std()),    4),
        }

    # ── Categorical value counts ───────────────
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        vc = df[col].value_counts().head(top_n_categories)
        distributions[str(col)] = {
            "type":         "categorical",
            "labels":       [str(label) for label in vc.index.tolist()],
            "values":       [int(v) for v in vc.values.tolist()],
            "unique_count": int(df[col].nunique()),
        }

    return distributions


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def run_eda(
    df: pd.DataFrame,
    *,
    include_summary:       bool = True,
    include_correlation:   bool = True,
    include_distributions: bool = True,
    correlation_method:    str  = "pearson",
    histogram_bins:        int  = 20,
    top_n_categories:      int  = 10,
) -> Dict[str, Any]:
    """
    Run exploratory data analysis on *df* and return a JSON-ready dict.

    Return value
    ------------
    {
        "shape":         {"rows": int, "columns": int},
        "columns":       [{"name": str, "dtype": str}, ...],
        "summary_stats": { col: { stat: value, ... }, ... },
        "correlation":   { col_a: { col_b: float|None, ... }, ... },
        "distributions": { col: { "type": ..., ... }, ... },
    }

    Parameters
    ----------
    df : pd.DataFrame
        The (ideally cleaned) input dataframe.
    include_summary : bool
        Generate per-column summary statistics.
    include_correlation : bool
        Compute the numeric correlation matrix.
    include_distributions : bool
        Generate histogram / value-count distribution data.
    correlation_method : str
        ``'pearson'``, ``'spearman'``, or ``'kendall'``.
        FIX-2: raises ValueError immediately for any other value.
    histogram_bins : int
        Number of bins for numeric histograms.
    top_n_categories : int
        Max categories returned for categorical distributions.

    Returns
    -------
    dict
        JSON-serialisable report.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If *correlation_method* is not one of the three valid options.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    # FIX-2: validate upfront so the caller gets a clear error before
    # any expensive computation runs
    if include_correlation and correlation_method not in _VALID_CORR_METHODS:
        raise ValueError(
            f"correlation_method must be one of {set(_VALID_CORR_METHODS)}, "
            f"got '{correlation_method}'"
        )

    report: Dict[str, Any] = {
        "shape": {
            "rows":    int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": [
            {"name": str(col), "dtype": str(dtype)}
            for col, dtype in df.dtypes.items()
        ],
    }

    # 1. Summary statistics
    if include_summary:
        try:
            report["summary_stats"] = _summary_stats(df)
        except Exception:
            logger.exception("Error generating summary statistics")
            report["summary_stats"] = {}

    # 2. Correlation matrix
    if include_correlation:
        try:
            report["correlation"] = _correlation_matrix(
                df, method=correlation_method,
            )
        except Exception:
            logger.exception("Error computing correlation matrix")
            report["correlation"] = {}

    # 3. Distribution data
    if include_distributions:
        try:
            report["distributions"] = _distribution_data(
                df,
                num_bins=histogram_bins,
                top_n_categories=top_n_categories,
            )
        except Exception:
            logger.exception("Error generating distribution data")
            report["distributions"] = {}

    logger.info(
        "EDA complete — %d rows × %d cols, %d summary entries, "
        "%d correlations, %d distributions",
        df.shape[0],
        df.shape[1],
        len(report.get("summary_stats",  {})),
        len(report.get("correlation",    {})),
        len(report.get("distributions",  {})),
    )

    return report


# ═══════════════════════════════════════════════
# Self-tests  (python eda_engine.py)
# ═══════════════════════════════════════════════

def _run_tests() -> None:
    print("\n── eda_engine.py self-tests ──")

    # Shared fixture — mixed types, NaNs, duplicates already removed
    df = pd.DataFrame({
        "age":    [25, 30, 35, 40, 45, 50, np.nan],
        "salary": [30_000, 45_000, 55_000, 70_000, 85_000, 100_000, 60_000],
        "score":  [0.72, 0.81, 0.68, 0.91, 0.77, 0.85, 0.74],
        "dept":   ["eng", "eng", "mkt", "eng", "hr", "mkt", "hr"],
        "city":   ["NY", "LA", "NY", "SF", "NY", "LA", "SF"],
    })

    # ── 1. Full run — all sections ─────────────────────────────────────────────
    report = run_eda(df)
    assert "shape"         in report
    assert "columns"       in report
    assert "summary_stats" in report
    assert "correlation"   in report
    assert "distributions" in report
    assert report["shape"]["rows"]    == 7
    assert report["shape"]["columns"] == 5
    print(f"✓ Full run — shape={report['shape']}, "
          f"summary_keys={len(report['summary_stats'])}")

    # ── 2. Summary stats — numeric column ─────────────────────────────────────
    age_stats = report["summary_stats"]["age"]
    assert age_stats["missing"] == 1
    assert age_stats["count"]   == 6.0
    assert "mean"  in age_stats
    assert "50%"   in age_stats
    print(f"✓ Summary stats — age: count={age_stats['count']}, "
          f"missing={age_stats['missing']}")

    # ── 3. Summary stats — categorical column ─────────────────────────────────
    dept_stats = report["summary_stats"]["dept"]
    assert dept_stats["missing"] == 0
    assert "unique" in dept_stats or "count" in dept_stats
    print(f"✓ Summary stats — dept: {dept_stats}")

    # ── 4. Correlation matrix — pearson ───────────────────────────────────────
    assert "age"    in report["correlation"]
    assert "salary" in report["correlation"]
    age_salary_corr = report["correlation"]["age"]["salary"]
    assert age_salary_corr is not None
    assert -1.0 <= age_salary_corr <= 1.0
    # Self-correlation must be 1.0
    assert report["correlation"]["age"]["age"] == 1.0
    print(f"✓ Correlation pearson — age↔salary={age_salary_corr:.4f}")

    # ── 5. FIX-2: invalid correlation_method raises ValueError ────────────────
    try:
        run_eda(df, correlation_method="cosine")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cosine" in str(e)
    print("✓ FIX-2 invalid correlation_method raises ValueError")

    # ── 6. FIX-2: _correlation_matrix direct call also validates ─────────────
    try:
        _correlation_matrix(df, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)
    print("✓ FIX-2 _correlation_matrix direct call also validates")

    # ── 7. Correlation — spearman variant ────────────────────────────────────
    report_s = run_eda(df, include_summary=False,
                       include_distributions=False,
                       correlation_method="spearman")
    assert "correlation" in report_s
    assert "age" in report_s["correlation"]
    print("✓ Spearman correlation runs without error")

    # ── 8. Distributions — numeric histogram ─────────────────────────────────
    age_dist = report["distributions"]["age"]
    assert age_dist["type"]   == "numeric"
    assert len(age_dist["bins"]) == 21   # 20 bins → 21 edges
    assert len(age_dist["counts"]) == 20
    assert "mean" in age_dist and "median" in age_dist
    print(f"✓ Numeric distribution — age bins={len(age_dist['counts'])}")

    # ── 9. Distributions — categorical value counts ───────────────────────────
    dept_dist = report["distributions"]["dept"]
    assert dept_dist["type"]          == "categorical"
    assert len(dept_dist["labels"])   == 3          # eng, mkt, hr
    assert dept_dist["unique_count"]  == 3
    assert dept_dist["values"][0]     == 3          # "eng" appears 3 times
    print(f"✓ Categorical distribution — dept top labels: {dept_dist['labels']}")

    # ── 10. No numeric columns — correlation returns {} ───────────────────────
    df_cat = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    report_cat = run_eda(df_cat)
    assert report_cat["correlation"] == {}
    print("✓ No numeric cols → correlation returns {}")

    # ── 11. Single numeric column — correlation returns {} ────────────────────
    df_one = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
    report_one = run_eda(df_one)
    assert report_one["correlation"] == {}
    print("✓ Single numeric col → correlation returns {}")

    # ── 12. TypeError on non-DataFrame input ──────────────────────────────────
    try:
        run_eda([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "DataFrame" in str(e)
    print("✓ TypeError raised for non-DataFrame input")

    # ── 13. Selective sections — only distributions ───────────────────────────
    report_dist = run_eda(
        df,
        include_summary=False,
        include_correlation=False,
        include_distributions=True,
    )
    assert "summary_stats" not in report_dist
    assert "correlation"   not in report_dist
    assert "distributions" in     report_dist
    print("✓ Selective sections — only distributions requested")

    # ── 14. top_n_categories respected ───────────────────────────────────────
    df_many = pd.DataFrame({"cat": list("abcdefghijklmnopqrst")})  # 20 values
    report_n = run_eda(df_many, top_n_categories=5,
                       include_summary=False, include_correlation=False)
    assert len(report_n["distributions"]["cat"]["labels"]) == 5
    print("✓ top_n_categories=5 respected")

    # ── 15. histogram_bins respected ─────────────────────────────────────────
    df_num = pd.DataFrame({"v": range(100)})
    report_b = run_eda(df_num, histogram_bins=10,
                       include_summary=False, include_correlation=False)
    assert len(report_b["distributions"]["v"]["counts"]) == 10
    assert len(report_b["distributions"]["v"]["bins"])   == 11
    print("✓ histogram_bins=10 respected")

    # ── 16. Empty numeric series → distribution handled ───────────────────────
    df_allnan = pd.DataFrame({"x": [np.nan, np.nan]})
    report_nan = run_eda(df_allnan, include_correlation=False)
    assert report_nan["distributions"]["x"]["bins"]   == []
    assert report_nan["distributions"]["x"]["counts"] == []
    print("✓ All-NaN numeric column → empty bins/counts in distribution")

    # ── 17. Datetime column in summary stats ──────────────────────────────────
    df_dt = pd.DataFrame({
        "ts": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"]),
        "v":  [1, 2, 3],
    })
    report_dt = run_eda(df_dt, include_correlation=False,
                        include_distributions=False)
    ts_stats = report_dt["summary_stats"]["ts"]
    assert "min"   in ts_stats
    assert "max"   in ts_stats
    assert "count" in ts_stats
    print(f"✓ Datetime column in summary stats: min={ts_stats['min']}")

    # ── 18. Correlation NaN for constant column ───────────────────────────────
    df_const = pd.DataFrame({"x": [1, 2, 3], "c": [5, 5, 5]})
    report_const = run_eda(df_const, include_summary=False,
                           include_distributions=False)
    # "c" correlates with itself as 1.0, with "x" as NaN (zero variance)
    c_x_corr = report_const["correlation"]["c"]["x"]
    assert c_x_corr is None, f"Expected None for const↔other, got {c_x_corr}"
    print("✓ Constant column correlation is None (not crash)")

    print("\n✓ All eda_engine.py self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()