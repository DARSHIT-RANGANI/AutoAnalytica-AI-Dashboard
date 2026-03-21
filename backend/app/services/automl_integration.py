"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  automl_integration.py  — v1.0                                              ║
║  Production-grade modular pipeline integration layer                         ║
║                                                                              ║
║  Architecture                                                                ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  RAW DATA → feature_transform                                                ║
║           → split_data (train / val / test)                                  ║
║           → model_search (candidate + RL tuning)                             ║
║           → cross_validate                                                   ║
║           → build_ensemble                                                   ║
║           → distill_model (optional, large datasets)                         ║
║           → validate_model                                                   ║
║           → run_full_pipeline (orchestrator)                                 ║
║                                                                              ║
║  Agent hooks (called by run_automl_with_agents):                             ║
║    • rl_agent.py      — state-aware arm selection                            ║
║    • meta_model.py    — cross-run pattern recognition                        ║
║    • retrain_model.py — drift / degradation detection                        ║
║    • agent_system.py  — step orchestration + logging                         ║
║                                                                              ║
║  All imports from automl_service.py are lazy (inside functions) to           ║
║  avoid the circular-import that would occur at module load time.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION — ensures sibling service modules are importable regardless
# of the working directory (FastAPI from backend/, direct execution, pytest).
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
import pathlib as _pathlib

def _add_services_to_path() -> None:
    """Add app/services/ to sys.path if not already present."""
    _this_dir = _pathlib.Path(__file__).resolve().parent
    _str_dir  = str(_this_dir)
    if _str_dir not in _sys.path:
        _sys.path.insert(0, _str_dir)

_add_services_to_path()
# ─────────────────────────────────────────────────────────────────────────────


import json
import logging
import math
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STATE DIRECTORIES  (created at import time)
# ─────────────────────────────────────────────────────────────────────────────

_RL_STATE_DIR      = Path("rl_state")
_META_STATE_DIR    = Path("meta_state")
_RETRAIN_STATE_DIR = Path("retrain_state")
_AGENT_STATE_DIR   = Path("agent_system_state")
_PIPELINE_LOG_DIR  = Path("pipeline_logs")

for _d in [_RL_STATE_DIR, _META_STATE_DIR, _RETRAIN_STATE_DIR,
           _AGENT_STATE_DIR, _PIPELINE_LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    """UTC ISO-8601 timestamp for log entries."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json(obj: Any) -> Any:
    """Recursively make obj JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _safe_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return _safe_json(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    except Exception as exc:
        log.warning(f"[Integration] Could not load {path}: {exc}")
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        with open(path, "w") as fh:
            json.dump(_safe_json(data), fh, indent=2)
    except Exception as exc:
        log.warning(f"[Integration] Could not save {path}: {exc}")


def sanitize_for_json(obj: Any, _seen: set = None) -> Any:
    """
    Recursively sanitize a result dict for JSON serialization.

    Replaces every NaN / Inf / -Inf float with None and converts numpy
    scalars / arrays to plain Python types.

    Applied as the FINAL step in run_automl_with_agents() before the
    dict is returned to FastAPI, catching any NaN that was introduced by
    agent sub-systems (MetaModel, RetrainController, AgentSystem) after
    the core pipeline's convert_to_python() call.

    Uses a seen-set to handle any circular references safely.
    """
    if _seen is None:
        _seen = set()

    if isinstance(obj, dict):
        oid = id(obj)
        if oid in _seen:
            return {}
        _seen.add(oid)
        return {str(k): sanitize_for_json(v, _seen) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        result = [sanitize_for_json(v, _seen) for v in obj]
        return result

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist(), _seen)

    # Catch-all: any object that is not a JSON-native type
    # (e.g. sklearn estimator, XGBClassifier, Pipeline, datetime, Path)
    # is converted to its string representation rather than passed through,
    # which would cause json.dumps to raise TypeError.
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    # Everything else — convert to string so the response always serializes
    try:
        json.dumps(obj)   # test if already serializable
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE LOG ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class _PipelineLogger:
    """Lightweight stage logger — passed through every pipeline stage."""

    def __init__(self, run_id: str) -> None:
        self.run_id  = run_id
        self.stages: List[dict] = []

    def log_stage(self, name: str, status: str,
                  elapsed_s: float, details: dict | None = None) -> None:
        entry = {
            "stage":     name,
            "status":    status,        # "ok" | "skipped" | "error"
            "elapsed_s": round(elapsed_s, 3),
            "ts":        _ts(),
            "details":   details or {},
        }
        self.stages.append(entry)
        icon = "✓" if status == "ok" else ("⚠" if status == "skipped" else "✗")
        log.info(f"[Pipeline:{self.run_id[:8]}] {icon} {name}  ({elapsed_s:.2f}s)")

    def to_dict(self) -> List[dict]:
        return self.stages


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def feature_transform(
    df: pd.DataFrame,
    target_column: str,
    context_overrides: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Wrap automl_service.universal_cleaning() and dataset diagnostics.

    Parameters
    ----------
    df               : raw DataFrame (already loaded)
    target_column    : name of the prediction target column
    context_overrides: optional dict to tweak behaviour (unused in v1.0,
                       reserved for future per-run configuration)

    Returns
    -------
    (cleaned_df, transform_report)
    """
    # Lazy import to avoid circular dependency at module load
    from automl_service import universal_cleaning, _log_dataset_diagnostics

    t0 = time.perf_counter()
    log.info(f"[Transform] Input shape: {df.shape}")

    # ── Normalise column names (same logic as run_automl) ────────────────────
    import re
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^A-Za-z0-9_]", "", regex=True))
    target_column = re.sub(r"[^A-Za-z0-9_]", "",
                           target_column.strip().replace(" ", "_"))

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    # ── 26-step cleaning ─────────────────────────────────────────────────────
    cleaned_df = universal_cleaning(df, target_column)

    # ── Problem-type sniff (needed for diagnostics) ───────────────────────────
    from automl_service import _detect_problem_type
    y_tmp = cleaned_df[target_column]
    problem_type, _, _ = _detect_problem_type(y_tmp, cleaned_df.shape[1] - 1)

    # ── Dataset diagnostics ───────────────────────────────────────────────────
    diagnostics = _log_dataset_diagnostics(cleaned_df, target_column, problem_type)

    elapsed = time.perf_counter() - t0
    transform_report = {
        "original_shape":   list(df.shape),
        "cleaned_shape":    list(cleaned_df.shape),
        "target_column":    target_column,
        "problem_type":     problem_type,
        "diagnostics":      diagnostics,
        "elapsed_s":        round(elapsed, 3),
    }
    log.info(f"[Transform] Done in {elapsed:.2f}s  "
             f"{df.shape} → {cleaned_df.shape}")
    return cleaned_df, transform_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA MATERIALISATION + THREE-WAY SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    val_size:  float = 0.10,
    test_size: float = 0.20,
) -> Tuple[
    pd.DataFrame, pd.Series,   # X_train, y_train
    pd.DataFrame, pd.Series,   # X_val,   y_val
    pd.DataFrame, pd.Series,   # X_test,  y_test
    dict,                      # split_report
]:
    """
    Three-way stratified split: train (70%) / val (10%) / test (20%).

    The validation set is used for distillation and retrain checks;
    the test set is held out until final evaluation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    t0 = time.perf_counter()
    X  = df.drop(columns=[target_column])
    y  = df[target_column]

    # Encode features (OHE for categorical)
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Label-encode target for classification
    le: Optional[LabelEncoder] = None
    if problem_type == "classification":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    stratify_y = y if problem_type == "classification" else None

    # First cut: (train+val) vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=stratify_y,
    )

    # Second cut: train vs val (val_size is relative to remaining)
    val_frac = val_size / (1.0 - test_size)
    strat_tv = y_tv if problem_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac,
        random_state=42,
        shuffle=True,
        stratify=strat_tv,
    )

    elapsed = time.perf_counter() - t0
    split_report = {
        "n_total":  len(df),
        "n_train":  len(X_train),
        "n_val":    len(X_val),
        "n_test":   len(X_test),
        "val_size": val_size,
        "test_size":test_size,
        "label_encoder_classes": le.classes_.tolist() if le else None,
        "elapsed_s": round(elapsed, 3),
    }
    log.info(f"[Split] total={len(df)}  "
             f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, split_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MODEL ARCHITECTURE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def model_search(
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    problem_type: str,
    tier:         int,
    small_dataset: bool = False,
    context_overrides: dict | None = None,
) -> Tuple[dict, dict]:
    """
    Build and prioritise candidate model pool.

    Wraps:
      _build_cls_candidates / _build_reg_candidates
      _apply_meta_learning_priority
      detect_data_leakage

    Returns
    -------
    (candidates_dict, search_report)
    """
    from automl_service import (
        _build_cls_candidates,
        _build_reg_candidates,
        _apply_meta_learning_priority,
        detect_data_leakage,
        _get_tier,
    )

    t0 = time.perf_counter()
    n_rows, n_feats = X_train.shape

    # ── Leakage guard ────────────────────────────────────────────────────────
    X_clean, removed, leakage = detect_data_leakage(X_train, y_train, threshold=0.90)

    # ── Build candidate pool ──────────────────────────────────────────────────
    if problem_type == "classification":
        candidates = _build_cls_candidates(tier, small_dataset=small_dataset)
    else:
        low_feature_mode = n_feats <= 3
        candidates = _build_reg_candidates(tier,
                                           low_feature_mode=low_feature_mode,
                                           small_dataset=small_dataset)

    # ── Meta-learning prioritisation ──────────────────────────────────────────
    candidates = _apply_meta_learning_priority(
        candidates, n_rows=n_rows, n_features=n_feats,
        problem_type=problem_type,
    )

    elapsed = time.perf_counter() - t0
    search_report = {
        "n_candidates":    len(candidates),
        "candidate_names": list(candidates.keys()),
        "tier":            tier,
        "small_dataset":   small_dataset,
        "leakage_detected":leakage,
        "removed_features":removed,
        "elapsed_s":       round(elapsed, 3),
    }
    log.info(f"[ModelSearch] {len(candidates)} candidates built in {elapsed:.2f}s")
    return candidates, search_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CROSS VALIDATION (RL-Bandit driven)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    candidates:   dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    problem_type: str,
    tier:         int,
    n_folds:      int = 3,
    rl_state:     dict | None = None,
) -> Tuple[Any, str, dict, dict, dict, list, dict]:
    """
    Run the multi-seed RL pipeline over the candidate pool.

    Wraps: _run_rl_multi_seed (from automl_service)

    Returns
    -------
    (best_model, best_name, cv_scores, cv_std_scores,
     best_params_all, agent_history, bandit_stats)
    """
    from automl_service import (
        _run_rl_multi_seed,
        _build_rl_state,
        auto_feature_selection,
    )
    from sklearn.model_selection import StratifiedKFold, KFold

    t0 = time.perf_counter()

    # Feature selection before CV
    n_classes = int(y_train.nunique()) if problem_type == "classification" else 1
    X_fs, _, fs_report = auto_feature_selection(
        X_train, X_train,           # X_test = X_train (intra-train FS)
        y_train, problem_type,
        n_classes=n_classes, tier=tier,
    )

    # Build RL state context
    if rl_state is None:
        rl_state = _build_rl_state(
            X_fs, y_train,
            problem_type=problem_type,
            tier=tier,
            n_candidates=len(candidates),
        )

    scoring = ("f1_macro" if problem_type == "classification" and n_classes > 2
               else ("f1"  if problem_type == "classification" else "r2"))

    # Dummy X_test (not used by _run_rl_multi_seed for scoring, only for final fit)
    X_dummy = X_fs.copy()
    y_dummy = y_train.copy()

    (best_name, best_model,
     cv_scores, cv_std_scores,
     best_params_all, agent_history,
     bandit_stats) = _run_rl_multi_seed(
        X_train=X_fs,
        X_test=X_dummy,
        y_train=y_train,
        y_test=y_dummy,
        candidates=candidates,
        problem_type=problem_type,
        tier=tier,
        scoring=scoring,
        n_folds=n_folds,
        rl_state=rl_state,
    )

    elapsed = time.perf_counter() - t0
    log.info(f"[CrossValidate] Winner: {best_name}  "
             f"cv={cv_scores.get(best_name, 0):.4f}  "
             f"({elapsed:.2f}s)")

    return (best_model, best_name, cv_scores, cv_std_scores,
            best_params_all, agent_history, bandit_stats)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ENSEMBLE CREATION
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble(
    cv_scores:      dict,
    candidates:     dict,
    best_params_all: dict,
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    problem_type:   str,
    top_n:          int = 3,
) -> Tuple[Any, str, dict]:
    """
    Wraps _build_stacking_model and evaluates whether stacking beats
    the best single-model CV score.

    Returns
    -------
    (ensemble_model_or_None, ensemble_name, ensemble_report)
    """
    from automl_service import _build_stacking_model
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    import numpy as np

    t0 = time.perf_counter()

    stack_model, base_names = _build_stacking_model(
        best_scores=cv_scores,
        candidates=candidates,
        best_params_all=best_params_all,
        problem_type=problem_type,
        top_n=top_n,
    )

    ensemble_report: dict = {
        "attempted": stack_model is not None,
        "base_models": base_names,
        "promoted": False,
        "stack_cv_score": None,
        "elapsed_s": 0.0,
    }

    if stack_model is None:
        log.info("[Ensemble] Skipped (fewer than 2 viable base models).")
        return None, "N/A", ensemble_report

    try:
        stack_model.fit(X_train, y_train)
        scoring = ("f1_macro" if problem_type == "classification"
                               and y_train.nunique() > 2
                   else ("f1" if problem_type == "classification" else "r2"))
        cv_klass = (StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    if problem_type == "classification"
                    else KFold(n_splits=3, shuffle=True, random_state=42))

        stack_cv   = cross_val_score(stack_model, X_train, y_train,
                                     cv=cv_klass, scoring=scoring, n_jobs=-1)
        stack_score = float(stack_cv.mean())
        stack_std   = float(stack_cv.std())

        best_single = max(cv_scores.values()) if cv_scores else 0.0
        promoted    = stack_score > best_single
        stacking_name = f"Stacking({'+'.join(base_names)})"

        ensemble_report.update({
            "stack_cv_score": round(stack_score, 4),
            "stack_cv_std":   round(stack_std,   4),
            "best_single_cv": round(best_single, 4),
            "promoted":       promoted,
        })
        log.info(f"[Ensemble] stack_cv={stack_score:.4f}  "
                 f"best_single={best_single:.4f}  promoted={promoted}")

        ensemble_report["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return stack_model, stacking_name, ensemble_report

    except Exception as exc:
        log.warning(f"[Ensemble] Fit/CV failed: {exc}")
        ensemble_report["error"] = str(exc)
        ensemble_report["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return None, "N/A", ensemble_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — OPTIONAL DISTILLATION
# ─────────────────────────────────────────────────────────────────────────────

_DISTILL_ROW_THRESHOLD = 50_000   # only distil on large datasets


def distill_model(
    teacher_model: Any,
    X_val:         pd.DataFrame,
    y_val:         pd.Series,
    problem_type:  str,
    n_rows:        int,
) -> Tuple[Any, dict]:
    """
    Lightweight knowledge distillation: train a fast student model on
    soft labels produced by the teacher.

    Strategy
    ────────
    • n_rows < _DISTILL_ROW_THRESHOLD → skip (returns teacher unchanged)
    • classification  → student = LogisticRegression trained on predict_proba
    • regression      → student = Ridge trained on teacher predictions

    This intentionally stays simple — heavy distillation (KL-divergence
    temperature scaling etc.) is delegated to a future dedicated module.

    Returns
    -------
    (student_or_teacher, distill_report)
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    t0 = time.perf_counter()
    distill_report: dict = {
        "applied": False,
        "reason":  "",
        "student": None,
        "elapsed_s": 0.0,
    }

    if n_rows < _DISTILL_ROW_THRESHOLD:
        distill_report["reason"] = (
            f"Skipped — dataset ({n_rows:,} rows) below "
            f"threshold ({_DISTILL_ROW_THRESHOLD:,})"
        )
        log.info(f"[Distill] {distill_report['reason']}")
        return teacher_model, distill_report

    try:
        if problem_type == "classification":
            if hasattr(teacher_model, "predict_proba"):
                soft_labels = teacher_model.predict_proba(X_val)
                # Student trained on soft label argmax (hard variant)
                hard_labels = soft_labels.argmax(axis=1)
                student = LogisticRegression(max_iter=500, random_state=42)
                student.fit(X_val, hard_labels)
                distill_report.update({
                    "applied": True,
                    "student": "LogisticRegression",
                    "reason":  "Large dataset — distilled to LR student.",
                })
                log.info("[Distill] Classification student trained.")
                return student, distill_report
            else:
                distill_report["reason"] = "Teacher has no predict_proba — skipped."
        else:
            teacher_preds = teacher_model.predict(X_val)
            student = Ridge(alpha=1.0)
            student.fit(X_val, teacher_preds)
            distill_report.update({
                "applied": True,
                "student": "Ridge",
                "reason":  "Large dataset — distilled to Ridge student.",
            })
            log.info("[Distill] Regression student trained.")
            return student, distill_report

    except Exception as exc:
        distill_report["reason"] = f"Distillation failed: {exc}"
        log.warning(f"[Distill] {distill_report['reason']}")

    distill_report["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return teacher_model, distill_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — VALIDATION (hold-out test set)
# ─────────────────────────────────────────────────────────────────────────────

def validate_model(
    model:        Any,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    problem_type: str,
    fs_report:    dict | None = None,
) -> dict:
    """
    Comprehensive hold-out evaluation.

    Wraps:
      detect_overfitting, calculate_confidence, _get_feature_importance,
      _get_shap_explanation, _quality_commentary

    Returns
    -------
    validation_report (dict)  — matches the schema expected by run_automl()
    """
    from automl_service import (
        detect_overfitting,
        calculate_confidence,
        _get_feature_importance,
        _get_shap_explanation,
        _quality_commentary,
        _safe_float,
        convert_to_python,
    )
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix,
        r2_score, mean_absolute_error, mean_squared_error,
    )
    import numpy as np

    t0 = time.perf_counter()
    n_total = len(X_train) + len(X_test)

    # ── Predictions ───────────────────────────────────────────────────────────
    preds        = model.predict(X_test)
    train_preds  = model.predict(X_train)

    report: dict = {
        "n_test":  len(X_test),
        "n_train": len(X_train),
        "problem_type": problem_type,
    }

    if problem_type == "classification":
        n_classes     = int(y_test.nunique())
        test_acc      = float(accuracy_score(y_test, preds))
        train_acc     = float(accuracy_score(y_train, train_preds))
        f1_val        = float(f1_score(y_test, preds,
                                        average="macro", zero_division=0))
        cm            = confusion_matrix(y_test, preds).tolist()
        overfit       = detect_overfitting(train_acc, test_acc)
        confidence    = calculate_confidence(test_acc, 0.0, n_total, overfit)

        roc_auc_val = None
        if n_classes == 2:
            try:
                proba = (model.predict_proba(X_test)[:, 1]
                         if hasattr(model, "predict_proba")
                         else model.decision_function(X_test))
                roc_auc_val = _safe_float(roc_auc_score(y_test, proba))
            except Exception:
                pass

        baseline_acc = float(y_train.value_counts(normalize=True).max())
        quality      = _quality_commentary(
            problem_type=problem_type,
            accuracy=test_acc, r2=None, mae=None,
            baseline_acc=baseline_acc, n_classes=n_classes,
        )
        report.update({
            "accuracy":        test_acc,
            "train_accuracy":  train_acc,
            "f1_macro":        f1_val,
            "roc_auc":         roc_auc_val,
            "confusion_matrix":convert_to_python(cm),
            "n_classes":       n_classes,
            "baseline_accuracy":baseline_acc,
            "overfitting":     overfit,
            "confidence_score":confidence["score"],
            "confidence_label":confidence["label"],
            "quality":         quality,
        })

    else:   # regression
        test_r2   = _safe_float(r2_score(y_test, preds))
        test_mae  = _safe_float(mean_absolute_error(y_test, preds))
        test_rmse = _safe_float(float(np.sqrt(mean_squared_error(y_test, preds))))
        train_r2  = _safe_float(r2_score(y_train, train_preds))
        overfit   = detect_overfitting(train_r2 or 0.0, test_r2 or 0.0)
        confidence = calculate_confidence(test_r2 or 0.0, 0.0, n_total, overfit)
        quality    = _quality_commentary(
            problem_type=problem_type,
            accuracy=None, r2=test_r2, mae=test_mae,
            baseline_acc=None, n_classes=1,
        )
        report.update({
            "R2":              test_r2,
            "MAE":             test_mae,
            "RMSE":            test_rmse,
            "train_R2":        train_r2,
            "overfitting":     overfit,
            "confidence_score":confidence["score"],
            "confidence_label":confidence["label"],
            "quality":         quality,
        })

    # ── Feature importance + SHAP ─────────────────────────────────────────────
    feat_imp = _get_feature_importance(model, X_test.columns.tolist(),
                                       fs_report=fs_report)
    shap_exp = _get_shap_explanation(model, X_train, X_test, shap_sample_n=500)

    report["feature_importance"]  = feat_imp
    report["shap_explanation"]    = shap_exp
    report["elapsed_s"]           = round(time.perf_counter() - t0, 3)

    log.info(f"[Validate] Done in {report['elapsed_s']:.2f}s  "
             f"quality={quality.get('rating','?')}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    file_path:         str,
    target_column:     str,
    time_budget:       Optional[float] = None,    # seconds; None = unlimited
    context_overrides: dict | None = None,
) -> dict:
    """
    End-to-end modular AutoML pipeline.

    Flow
    ────
    1. load_data
    2. feature_transform
    3. split_data  (train / val / test)
    4. model_search
    5. cross_validate  (RL-Bandit)
    6. build_ensemble
    7. distill_model  (optional)
    8. validate_model
    9. persist model + logs

    Returns
    -------
    {
      "model":                 trained model object,
      "model_name":            saved .pkl filename,
      "metrics":               validation_report,
      "pipeline_stage_logs":   list of stage entries,
      "intermediate_outputs":  per-stage details,
      "friendly_summary":      human-readable string,
    }
    """
    from automl_service import (
        _load_file, _get_tier, _save_model, convert_to_python,
    )

    run_id   = uuid.uuid4().hex
    plogger  = _PipelineLogger(run_id)
    wall_t0  = time.perf_counter()
    ctx      = context_overrides or {}

    intermediate: dict = {}

    def _budget_ok() -> bool:
        if time_budget is None:
            return True
        return (time.perf_counter() - wall_t0) < time_budget

    # ── 1. Load ───────────────────────────────────────────────────────────────
    t = time.perf_counter()
    fp = Path(file_path)
    if not fp.exists():
        # Try uploads sub-directory (FastAPI convention)
        fp = Path("uploads") / file_path
    if not fp.exists():
        return {"error": f"File not found: {file_path}"}

    raw = _load_file(fp)
    if isinstance(raw, dict):          # error dict from _load_file
        return raw
    df: pd.DataFrame = raw
    plogger.log_stage("load", "ok", time.perf_counter() - t,
                      {"rows": len(df), "cols": df.shape[1]})

    # ── 2. Transform ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    try:
        cleaned_df, transform_report = feature_transform(
            df, target_column, context_overrides=ctx)
        intermediate["transform"] = transform_report
        problem_type = transform_report["problem_type"]
        plogger.log_stage("feature_transform", "ok", time.perf_counter() - t,
                          {"shape": list(cleaned_df.shape)})
    except Exception as exc:
        plogger.log_stage("feature_transform", "error", time.perf_counter() - t,
                          {"error": str(exc)})
        return {"error": f"Feature transform failed: {exc}",
                "pipeline_stage_logs": plogger.to_dict()}

    # Normalise target_column (same regex as feature_transform does internally)
    import re
    target_column = re.sub(r"[^A-Za-z0-9_]", "",
                           target_column.strip().replace(" ", "_"))

    # ── 3. Split ──────────────────────────────────────────────────────────────
    t = time.perf_counter()
    try:
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         split_report) = split_data(cleaned_df, target_column, problem_type)
        intermediate["split"] = split_report
        n_total    = split_report["n_total"]
        tier       = _get_tier(n_total)
        small_data = n_total < 200
        plogger.log_stage("split_data", "ok", time.perf_counter() - t,
                          split_report)
    except Exception as exc:
        plogger.log_stage("split_data", "error", time.perf_counter() - t,
                          {"error": str(exc)})
        return {"error": f"Data split failed: {exc}",
                "pipeline_stage_logs": plogger.to_dict()}

    # ── 4. Model search ───────────────────────────────────────────────────────
    t = time.perf_counter()
    if not _budget_ok():
        log.warning("[Pipeline] Budget exhausted after split.")
    try:
        candidates, search_report = model_search(
            X_train, y_train, problem_type, tier,
            small_dataset=small_data, context_overrides=ctx)
        intermediate["model_search"] = search_report
        plogger.log_stage("model_search", "ok", time.perf_counter() - t,
                          {"n_candidates": len(candidates)})
    except Exception as exc:
        plogger.log_stage("model_search", "error", time.perf_counter() - t,
                          {"error": str(exc)})
        return {"error": f"Model search failed: {exc}",
                "pipeline_stage_logs": plogger.to_dict()}

    # ── 5. Cross validation (RL) ──────────────────────────────────────────────
    t = time.perf_counter()
    try:
        n_folds = 10 if small_data else 3
        (best_model, best_name,
         cv_scores, cv_std_scores,
         best_params_all, agent_history,
         bandit_stats) = cross_validate(
            candidates, X_train, y_train,
            problem_type, tier, n_folds=n_folds)
        intermediate["cross_validation"] = {
            "best_name": best_name,
            "cv_scores": {k: round(v, 4) for k, v in cv_scores.items()},
            "agent_history_length": len(agent_history),
        }
        plogger.log_stage("cross_validate", "ok", time.perf_counter() - t,
                          {"winner": best_name,
                           "cv_score": round(cv_scores.get(best_name, 0), 4)})
    except Exception as exc:
        plogger.log_stage("cross_validate", "error", time.perf_counter() - t,
                          {"error": str(exc)})
        return {"error": f"Cross validation failed: {exc}",
                "pipeline_stage_logs": plogger.to_dict()}

    if best_model is None:
        return {"error": "All candidates failed during cross validation.",
                "pipeline_stage_logs": plogger.to_dict()}

    # ── 6. Ensemble ───────────────────────────────────────────────────────────
    t = time.perf_counter()
    try:
        ensemble_model, ensemble_name, ensemble_report = build_ensemble(
            cv_scores, candidates, best_params_all,
            X_train, y_train, problem_type, top_n=3)
        intermediate["ensemble"] = ensemble_report

        # Promote ensemble if it beat the single-model winner
        if ensemble_report.get("promoted") and ensemble_model is not None:
            best_model = ensemble_model
            best_name  = ensemble_name
            log.info(f"[Pipeline] Ensemble promoted: {ensemble_name}")

        plogger.log_stage("build_ensemble", "ok", time.perf_counter() - t,
                          {"promoted": ensemble_report.get("promoted"),
                           "name": ensemble_name})
    except Exception as exc:
        log.warning(f"[Pipeline] Ensemble stage error (non-fatal): {exc}")
        plogger.log_stage("build_ensemble", "error", time.perf_counter() - t,
                          {"error": str(exc)})

    # ── 7. Distillation (optional) ────────────────────────────────────────────
    t = time.perf_counter()
    try:
        final_model, distill_report = distill_model(
            best_model, X_val, y_val, problem_type, n_total)
        intermediate["distillation"] = distill_report
        plogger.log_stage(
            "distill_model",
            "ok" if distill_report["applied"] else "skipped",
            time.perf_counter() - t,
            {"applied": distill_report["applied"]})
    except Exception as exc:
        log.warning(f"[Pipeline] Distillation error (non-fatal): {exc}")
        final_model = best_model
        plogger.log_stage("distill_model", "error", time.perf_counter() - t,
                          {"error": str(exc)})

    # ── 8. Validate (held-out test set) ───────────────────────────────────────
    t = time.perf_counter()
    try:
        val_report = validate_model(
            final_model, X_test, y_test,
            X_train, y_train, problem_type)
        intermediate["validation"] = val_report
        plogger.log_stage("validate_model", "ok", time.perf_counter() - t,
                          {"quality": val_report.get("quality", {}).get("rating")})
    except Exception as exc:
        plogger.log_stage("validate_model", "error", time.perf_counter() - t,
                          {"error": str(exc)})
        return {"error": f"Validation failed: {exc}",
                "pipeline_stage_logs": plogger.to_dict()}

    # ── 9. Persist ────────────────────────────────────────────────────────────
    model_fname = _save_model(final_model, X_test.columns.tolist(), problem_type)

    total_elapsed = round(time.perf_counter() - wall_t0, 2)
    quality       = val_report.get("quality", {})
    score_key     = "accuracy" if problem_type == "classification" else "R2"
    score_val     = val_report.get(score_key)

    friendly_summary = (
        f"[{_TIER_NAMES.get(tier, 'UNKNOWN')}: {n_total:,} rows]  "
        f"Best model: {best_name}.  "
        f"Test {score_key}: {score_val:.4f}.  "
        f"Quality: {quality.get('rating','?')} — {quality.get('summary','')}.  "
        f"Total time: {total_elapsed}s."
    )
    log.info(f"[Pipeline] {friendly_summary}")

    return convert_to_python({
        "run_id":                run_id,
        "model":                 model_fname,          # filename only — object is on disk
        "model_name":            model_fname,
        "best_model_name":       best_name,
        "metrics":               val_report,
        "cv_scores":             cv_scores,
        "cv_std_scores":         cv_std_scores,
        "best_params":           {k.replace("model__", ""): v
                                   for k, v in best_params_all.get(best_name, {}).items()},
        "agent_history":         agent_history,
        "bandit_stats":          bandit_stats,
        "pipeline_stage_logs":   plogger.to_dict(),
        "intermediate_outputs":  intermediate,
        "problem_type":          problem_type,
        "tier":                  tier,
        "tier_name":             _TIER_NAMES.get(tier, "UNKNOWN"),
        "total_elapsed_s":       total_elapsed,
        "friendly_summary":      friendly_summary,
    })


# Tier name lookup (mirrors automl_service constants)
_TIER_NAMES = {1: "SMALL", 2: "MEDIUM", 3: "LARGE", 4: "MASSIVE"}


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME STORE  (persistent JSON ledger for meta-model / retrain)
# ─────────────────────────────────────────────────────────────────────────────

_OUTCOME_LEDGER = _AGENT_STATE_DIR / "outcome_ledger.json"


def _load_ledger() -> list:
    data = _load_json(_OUTCOME_LEDGER)
    return data.get("records", []) if isinstance(data, dict) else []


def _append_ledger(record: dict) -> None:
    records = _load_ledger()
    records.append(record)
    # Keep last 1000 records
    if len(records) > 1000:
        records = records[-1000:]
    _save_json(_OUTCOME_LEDGER, {"records": records,
                                  "last_updated": _ts()})


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (imported by automl_service.py Section 13)
# ─────────────────────────────────────────────────────────────────────────────

def run_automl_with_agents(
    filename:          str,
    target_column:     str,
    time_budget:       Optional[float] = None,
    context_overrides: dict | None = None,
    save_after_run:    bool = True,
) -> dict:
    """
    Drop-in replacement for automl_service.run_automl().

    Runs the full modular pipeline and injects:
      • RL decision       — from rl_agent (if available)
      • meta insight      — from meta_model (if available)
      • retrain decision  — from retrain_model (if available)
      • agent system log  — from agent_system (if available)

    Returns the extended result dict expected by the FastAPI layer.
    """
    run_id = uuid.uuid4().hex
    log.info(f"[WithAgents] run_id={run_id}  file={filename}  target={target_column}")

    # ── Run core pipeline ─────────────────────────────────────────────────────
    file_path = str(Path("uploads") / filename)
    result    = run_full_pipeline(file_path, target_column,
                                  time_budget=time_budget,
                                  context_overrides=context_overrides)

    if "error" in result:
        return result

    metrics = result.get("metrics", {})

    # ── RL Agent hook ─────────────────────────────────────────────────────────
    rl_decision: dict = {"status": "not_integrated"}
    try:
        from rl_agent import RLAgent   # noqa: F401  (optional module)
        agent   = RLAgent.load_or_create()
        state   = _build_rl_state_from_result(result)
        action  = agent.choose_action(state)
        agent.update(state, action, reward=metrics.get("confidence_score", 0.0))
        if save_after_run:
            agent.save()
        rl_decision = {"status": "ok", "action": action, "state": state}
    except ImportError:
        rl_decision["status"] = "module_not_found"
    except Exception as exc:
        rl_decision["status"] = f"error: {exc}"
        log.warning(f"[WithAgents] RL Agent hook failed: {exc}")

    # ── Meta model hook ───────────────────────────────────────────────────────
    meta_insight: dict = {"status": "not_integrated"}
    try:
        from meta_model import MetaModel  # noqa: F401
        mm      = MetaModel.load_or_create()
        insight = mm.predict(metrics)
        if save_after_run:
            mm.save()
        meta_insight = {"status": "ok", "insight": insight}
    except ImportError:
        meta_insight["status"] = "module_not_found"
    except Exception as exc:
        meta_insight["status"] = f"error: {exc}"
        log.warning(f"[WithAgents] Meta Model hook failed: {exc}")

    # ── Retrain decision hook ─────────────────────────────────────────────────
    retrain_decision: dict = {"status": "not_integrated", "should_retrain": False}
    try:
        from retrain_model import RetrainController  # noqa: F401
        rc = RetrainController.load_or_create()
        should, reason = rc.should_retrain(metrics)
        retrain_decision = {"status": "ok", "should_retrain": should, "reason": reason}
        if should:
            log.info(f"[WithAgents] Retrain recommended: {reason}")
        if save_after_run:
            rc.save()
    except ImportError:
        retrain_decision["status"] = "module_not_found"
    except Exception as exc:
        retrain_decision["status"] = f"error: {exc}"
        log.warning(f"[WithAgents] Retrain hook failed: {exc}")

    # ── Agent system hook ─────────────────────────────────────────────────────
    agent_system_log: list = []
    try:
        from agent_system import AgentSystem  # noqa: F401
        asys = AgentSystem.load_or_create()
        asys.record_run(run_id, result)
        agent_system_log = asys.get_recent_log(n=20)
        if save_after_run:
            asys.save()
    except ImportError:
        agent_system_log = [{"note": "agent_system module not found"}]
    except Exception as exc:
        agent_system_log = [{"error": str(exc)}]
        log.warning(f"[WithAgents] Agent system hook failed: {exc}")

    # ── Persist outcome to ledger ─────────────────────────────────────────────
    if save_after_run:
        _append_ledger({
            "run_id":    run_id,
            "ts":        _ts(),
            "filename":  filename,
            "target":    target_column,
            "best_model":result.get("best_model_name"),
            "metrics":   {k: v for k, v in metrics.items()
                          if isinstance(v, (int, float, bool, str, type(None)))},
        })

    # ── Build final extended response ─────────────────────────────────────────
    extended = dict(result)
    extended.update({
        "decision":          rl_decision.get("action"),
        "rl_decision":       rl_decision,
        "meta_insight":      meta_insight.get("insight") if meta_insight.get("status") == "ok" else None,
        "meta_insight_full": meta_insight,
        "retrain_decision":  retrain_decision,
        "agent_system_log":  agent_system_log,
        # Flatten key metrics to top level for quick API consumption
        "model_metrics":     metrics,
        "prediction":        {
            "problem_type":  result.get("problem_type"),
            "best_model":    result.get("best_model_name"),
            "score":         (metrics.get("accuracy") or metrics.get("R2")),
            "confidence":    metrics.get("confidence_label"),
        },
    })
    return sanitize_for_json(extended)


def _build_rl_state_from_result(result: dict) -> dict:
    """Convert a pipeline result dict into the minimal state vector."""
    metrics = result.get("metrics", {})
    interm  = result.get("intermediate_outputs", {})
    split   = interm.get("split", {})
    search  = interm.get("model_search", {})
    return {
        "n_rows":       split.get("n_train", 0),
        "n_features":   search.get("n_candidates", 0),
        "problem_type": result.get("problem_type", "unknown"),
        "tier":         result.get("tier", 1),
        "cv_score":     max(result.get("cv_scores", {}).values(), default=0.0),
        "confidence":   metrics.get("confidence_score", 0.0),
        "overfitting":  int(metrics.get("overfitting", False)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATE MANAGEMENT  (public stubs used by automl_service.py Section 13)
# ─────────────────────────────────────────────────────────────────────────────

def record_outcome(
    run_id:  str,
    metrics: dict,
    context: dict | None = None,
) -> dict:
    """
    Persist a labelled outcome to the ledger for offline meta-model training.

    Can be called by the FastAPI route after the user rates predictions.
    """
    record = {
        "run_id":  run_id,
        "ts":      _ts(),
        "metrics": _safe_json(metrics),
        "context": _safe_json(context or {}),
    }
    _append_ledger(record)
    log.info(f"[Outcome] Recorded run_id={run_id}")
    return {"status": "ok", "run_id": run_id, "ts": record["ts"]}


def agent_status() -> dict:
    """Return availability status of each optional AI module."""
    modules = {
        "rl_agent":     "rl_agent.py",
        "meta_model":   "meta_model.py",
        "retrain_model":"retrain_model.py",
        "agent_system": "agent_system.py",
    }
    status: dict = {"agents_available": True, "modules": {}}
    for key, mod_name in modules.items():
        try:
            __import__(mod_name.replace(".py", ""))
            status["modules"][key] = "available"
        except ImportError:
            status["modules"][key] = "not_installed"
            status["agents_available"] = False

    records = _load_ledger()
    status["outcome_ledger"] = {
        "n_records":    len(records),
        "last_updated": records[-1]["ts"] if records else None,
    }
    return status


def save_agents() -> dict:
    """
    Flush in-memory agent state to disk.
    Delegates to each module's .save() if available.
    """
    saved = []
    skipped = []
    for mod_name, cls_name in [("rl_agent", "RLAgent"),
                                ("meta_model", "MetaModel"),
                                ("retrain_model", "RetrainController"),
                                ("agent_system", "AgentSystem")]:
        try:
            mod = __import__(mod_name)
            cls = getattr(mod, cls_name)
            instance = cls.load_or_create()
            instance.save()
            saved.append(mod_name)
        except ImportError:
            skipped.append(mod_name)
        except Exception as exc:
            log.warning(f"[SaveAgents] {mod_name} save failed: {exc}")
            skipped.append(mod_name)

    return {"status": "ok", "saved": saved, "skipped": skipped, "ts": _ts()}


def reset_agent_system() -> dict:
    """
    Wipe all persistent agent state (useful for testing / fresh deployments).
    Does NOT delete the outcome ledger — those are valuable training labels.
    """
    wiped = []
    for state_dir in [_RL_STATE_DIR, _META_STATE_DIR,
                      _RETRAIN_STATE_DIR, _AGENT_STATE_DIR]:
        try:
            for f in state_dir.glob("*.json"):
                if f.name != "outcome_ledger.json":
                    f.unlink()
                    wiped.append(str(f))
        except Exception as exc:
            log.warning(f"[Reset] Could not wipe {state_dir}: {exc}")

    log.info(f"[Reset] Wiped {len(wiped)} state file(s).")
    return {"status": "ok", "wiped_files": wiped, "ts": _ts()}