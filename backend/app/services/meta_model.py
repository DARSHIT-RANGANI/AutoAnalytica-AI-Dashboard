"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  meta_model.py  — v2.0  (AutoAnalytica v5.5)                               ║
║                                                                              ║
║  Cross-Run Pattern Recognition & Insight Generation                          ║
║                                                                              ║
║  Role in the stack                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Layer 0 (automl_service.py)  : per-run model selection (MAB)               ║
║  Layer 1 (rl_agent.py)        : cross-run pipeline action selection (Q-table)║
║  Layer 2 (meta_model.py)      : cross-run pattern recognition → insights    ║
║                                                                              ║
║  What it learns                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  From the outcome ledger (populated by record_outcome / run_automl_with_    ║
║  agents) it learns:                                                          ║
║    • Which models win most often for each dataset profile                    ║
║    • Expected CV score range per profile                                     ║
║    • Which risk conditions (overfitting, leakage, imbalance) co-occur       ║
║    • How often stacking outperforms the single-model winner                  ║
║                                                                              ║
║  Two-phase learning                                                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Phase 1 — Heuristic (< MIN_RECORDS runs):                                  ║
║    Lookup-table of profile → aggregated stats from past runs.               ║
║    Falls back to rule-based recommendations when the table is empty.        ║
║                                                                              ║
║  Phase 2 — Fitted sklearn model (≥ MIN_RECORDS runs):                       ║
║    Trains a GradientBoostingRegressor on extracted meta-features to         ║
║    predict expected CV score.  Serialised as meta_state/meta_model.pkl.     ║
║                                                                              ║
║  v2.0 Changes                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  + RecommendationModel    : GradientBoosting replaces _make_recommendation  ║
║                             if-elif chain — learns which recommendation     ║
║                             category fits from meta-features                 ║
║  + ModelSuggestionPredictor: GradientBoosting replaces hardcoded model      ║
║                             name lists in cold-start insight                 ║
║  + AdaptiveRiskThreshold  : learns per-flag alert thresholds from           ║
║                             observed rate distributions — replaces 0.40     ║
║  + _adaptive_confidence() : combines count + score stability — replaces     ║
║                             hardcoded n>=10 / n>=3 thresholds               ║
║                                                                              ║
║  Persistence                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  meta_state/meta_table.json   — profile → aggregated stats                  ║
║  meta_state/meta_model.pkl    — fitted sklearn regressor (Phase 2)          ║
║  meta_state/meta_meta.json    — version, run counts, last fitted ts         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── Inline path bootstrap (no external module needed) ──────────────────────
import sys as _sys, pathlib as _pl
_sd = str(_pl.Path(__file__).resolve().parent)
if _sd not in _sys.path: _sys.path.insert(0, _sd)
del _sd
# ────────────────────────────────────────────────────────────────────────────
import json
import logging
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_META_STATE_DIR   = Path("meta_state")
_META_TABLE_PATH  = _META_STATE_DIR / "meta_table.json"
_META_MODEL_PATH  = _META_STATE_DIR / "meta_model.pkl"
_META_META_PATH   = _META_STATE_DIR / "meta_meta.json"

_META_STATE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum records before fitting a sklearn model
MIN_RECORDS_FOR_FIT = 20


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILE  (same bucketing strategy as rl_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_rows(n: float) -> str:
    if n < 200:      return "tiny"
    if n < 10_000:   return "small"
    if n < 100_000:  return "medium"
    return "large"


def _bucket_feats(n: float) -> str:
    if n < 10:   return "low"
    if n < 50:   return "medium"
    if n < 200:  return "high"
    return "very_high"


def _bucket_imbalance(r: Optional[float]) -> str:
    if r is None:  return "n/a"
    if r < 1.5:    return "balanced"
    if r < 4.0:    return "moderate"
    return "severe"


def _bucket_missing(pct: float) -> str:
    if pct < 5:    return "clean"
    if pct < 20:   return "moderate"
    return "heavy"


def dataset_profile(meta_features: Dict[str, Any]) -> str:
    """
    Convert a meta-feature dict into a discrete profile string.
    Used as the primary lookup key in MetaTable.

    Accepts keys from both dataset_diagnostics and the pipeline result dict.
    """
    n_rows    = float(meta_features.get("n_rows") or meta_features.get("rows", 0))
    n_feats   = float(meta_features.get("n_cols") or meta_features.get("features",
                 meta_features.get("original_features", 0)))
    missing   = float(meta_features.get("overall_missing_pct",
                 meta_features.get("missing_ratio", 0.0) * 100))
    imbalance = meta_features.get("imbalance_ratio")
    prob_type = meta_features.get("problem_type", "unknown")

    return (
        f"{prob_type}"
        f"|rows:{_bucket_rows(n_rows)}"
        f"|feat:{_bucket_feats(n_feats)}"
        f"|mis:{_bucket_missing(missing)}"
        f"|imb:{_bucket_imbalance(imbalance)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# META-FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_meta_features(pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull a flat, numeric meta-feature vector from a full pipeline result dict.

    Handles both the legacy run_automl() schema and the new
    run_automl_with_agents() / run_full_pipeline() schema.
    All values are cast to float / int / bool so they can be
    used directly as sklearn input rows.
    """
    perf  = pipeline_result.get("performance") or pipeline_result.get("metrics") or {}
    fs    = (perf.get("feature_selection")
             or pipeline_result.get("feature_selection") or {})
    diag  = (pipeline_result.get("dataset_diagnostics")
             or pipeline_result.get("intermediate_outputs", {}).get(
                 "transform", {}).get("diagnostics") or {})

    def _f(key, *fallbacks, default=0.0):
        """Safe float extractor with multiple fallback keys."""
        for k in (key, *fallbacks):
            v = perf.get(k) or pipeline_result.get(k) or diag.get(k)
            if v is not None:
                try:
                    fv = float(v)
                    return None if (math.isnan(fv) or math.isinf(fv)) else fv
                except (TypeError, ValueError):
                    pass
        return default

    def _b(key, *fallbacks) -> int:
        """Safe bool extractor → 0/1."""
        for k in (key, *fallbacks):
            v = perf.get(k) or pipeline_result.get(k) or diag.get(k)
            if v is not None:
                return int(bool(v))
        return 0

    # ── Dataset shape ─────────────────────────────────────────────────────────
    n_rows    = _f("n_rows", "rows", default=0.0)
    n_cols    = _f("n_cols", "features", default=0.0)
    missing   = _f("overall_missing_pct", default=0.0)
    n_classes = _f("n_classes", default=0.0)
    imb_ratio = _f("imbalance_ratio", default=1.0)
    skew      = _f("target_skew", default=0.0)

    # ── Performance signals ───────────────────────────────────────────────────
    cv_mean   = _f("cv_score_mean", default=0.0)
    cv_std    = _f("cv_score_std", default=0.0)
    test_sc   = _f("test_score", "accuracy", "R2", default=0.0)
    conf_sc   = _f("confidence_score", default=0.0)

    # ── Risk flags ────────────────────────────────────────────────────────────
    overfit   = _b("overfitting")
    leakage   = _b("leakage_detected")
    bl_alert  = int(bool((pipeline_result.get("baseline_alert") or {}).get(
        "triggered", False)))

    # ── Feature selection — read directly from fs dict first ─────────────────
    orig_f    = float(fs.get("original_features") or _f(
        "original_features", default=n_cols))
    final_f   = float(fs.get("final_features")    or _f(
        "final_features",    default=n_cols))
    pca_app   = int(bool(fs.get("pca_applied", False)))
    lda_app   = int(bool(fs.get("lda_applied", False)))
    fs_ratio  = round(final_f / max(orig_f, 1), 4)

    # ── Pipeline context ──────────────────────────────────────────────────────
    tier      = _f("scale_tier", default=1.0)
    n_folds   = _f("n_cv_folds", default=3.0)

    return {
        # Shape
        "n_rows":          n_rows,
        "n_cols":          n_cols,
        "missing_pct":     missing,
        "n_classes":       n_classes,
        "imbalance_ratio": imb_ratio,
        "target_skew":     skew,
        # Performance
        "cv_mean":         cv_mean,
        "cv_std":          cv_std,
        "test_score":      test_sc,
        "confidence_score":conf_sc,
        # Flags
        "overfitting":     overfit,
        "leakage":         leakage,
        "baseline_alert":  bl_alert,
        # Feature selection
        "orig_features":   orig_f,
        "final_features":  final_f,
        "fs_ratio":        fs_ratio,
        "pca_applied":     pca_app,
        "lda_applied":     lda_app,
        # Pipeline
        "tier":            tier,
        "n_cv_folds":      n_folds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# META TABLE  (profile → aggregated stats, Phase 1 memory)
# ─────────────────────────────────────────────────────────────────────────────

class MetaTable:
    """
    Lightweight lookup table: profile_key → aggregated run statistics.

    Each entry tracks:
      • n_runs            — how many runs matched this profile
      • cv_scores         — list of CV scores (capped at 200)
      • model_wins        — {model_name: win_count}
      • risk_counts       — {risk_flag: count}
      • stacking_wins     — times stacking beat single model
    """

    def __init__(self, data: Dict | None = None) -> None:
        self._table: Dict[str, Dict] = data or {}

    # ── Record a completed run ────────────────────────────────────────────────

    def record(self, profile: str, meta_features: Dict,
               best_model: str, stacking_won: bool) -> None:
        if profile not in self._table:
            self._table[profile] = {
                "n_runs":       0,
                "cv_scores":    [],
                "model_wins":   defaultdict(int),
                "risk_counts":  defaultdict(int),
                "stacking_wins":0,
            }
        entry = self._table[profile]
        entry["n_runs"] += 1
        cv = meta_features.get("cv_mean")
        if cv is not None and not math.isnan(cv):
            entry["cv_scores"].append(round(cv, 4))
            if len(entry["cv_scores"]) > 200:
                entry["cv_scores"] = entry["cv_scores"][-200:]

        entry["model_wins"][best_model] = (
            entry["model_wins"].get(best_model, 0) + 1)

        for flag in ("overfitting", "leakage", "baseline_alert"):
            if meta_features.get(flag, 0):
                entry["risk_counts"][flag] = (
                    entry["risk_counts"].get(flag, 0) + 1)

        if stacking_won:
            entry["stacking_wins"] += 1

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, profile: str) -> Optional[Dict]:
        return self._table.get(profile)

    def n_profiles(self) -> int:
        return len(self._table)

    def total_runs(self) -> int:
        return sum(e["n_runs"] for e in self._table.values())

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        # defaultdict → plain dict for JSON
        out = {}
        for profile, entry in self._table.items():
            out[profile] = {
                "n_runs":       entry["n_runs"],
                "cv_scores":    entry["cv_scores"],
                "model_wins":   dict(entry.get("model_wins", {})),
                "risk_counts":  dict(entry.get("risk_counts", {})),
                "stacking_wins":entry.get("stacking_wins", 0),
            }
        return out

    @classmethod
    def from_dict(cls, data: Dict) -> "MetaTable":
        inst = cls()
        for profile, entry in data.items():
            inst._table[profile] = {
                "n_runs":       entry.get("n_runs", 0),
                "cv_scores":    entry.get("cv_scores", []),
                "model_wins":   defaultdict(int, entry.get("model_wins", {})),
                "risk_counts":  defaultdict(int, entry.get("risk_counts", {})),
                "stacking_wins":entry.get("stacking_wins", 0),
            }
        return inst


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION MODEL  (replaces if-elif chain in _make_recommendation)
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationModel:
    """
    GradientBoosting classifier that predicts the most appropriate
    recommendation category from meta-features.

    Replaces the hardcoded 8-branch if-elif chain in _make_recommendation()
    once MIN_SAMPLES training examples are available.

    Training signal
    ───────────────
    Each historical record's meta-features are labelled with the dominant
    risk category derived from that run's outcome flags and scores.
    The model learns to recognise the same patterns from raw features —
    including non-obvious interactions that the rule chain cannot express.

    Falls back to the rule chain whenever the model is absent, returns
    low confidence, or sklearn is unavailable.
    """

    MIN_SAMPLES      = 15
    RETRAIN_EVERY    = 10
    CONFIDENCE_FLOOR = 0.35

    # Recommendation category labels
    CATEGORIES = [
        "leakage_priority",
        "baseline_priority",
        "overfit_small",
        "stacking_wins",
        "severe_imbalance",
        "high_cv_variance",
        "model_preference",
        "low_cv_historical",
        "default",
    ]

    def __init__(self) -> None:
        self._clf          = None
        self._next_retrain = self.MIN_SAMPLES

    # ── Public ────────────────────────────────────────────────────────────────

    def maybe_fit(self, records: List[Dict]) -> None:
        """Re-fit if enough records have arrived since last training session."""
        if len(records) >= self._next_retrain:
            self._fit(records)
            self._next_retrain = len(records) + self.RETRAIN_EVERY

    def predict(self, meta_features: Dict) -> Optional[str]:
        """
        Return the predicted recommendation category, or None when the
        model is absent or confidence falls below CONFIDENCE_FLOOR so
        the caller can safely fall back to the rule chain.
        """
        if self._clf is None:
            return None
        try:
            vec   = _meta_feature_vector(meta_features)
            proba = self._clf.predict_proba([vec])[0]
            best_p = float(proba.max())
            if best_p < self.CONFIDENCE_FLOOR:
                log.debug(
                    f"[RecommendationModel] Low confidence ({best_p:.3f}) "
                    f"— deferring to rule chain."
                )
                return None
            cat = str(self._clf.classes_[int(proba.argmax())])
            log.info(f"[RecommendationModel] → {cat}  (p={best_p:.3f})")
            return cat if cat in self.CATEGORIES else None
        except Exception as exc:
            log.warning(f"[RecommendationModel] predict failed: {exc}")
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _derive_label(self, mf: Dict,
                      stack_rate: float = 0.0,
                      cv_mean:    float = 0.0) -> str:
        """
        Derive the ground-truth recommendation category from outcome flags.
        Priority order mirrors the original rule chain — ensures the model
        learns the same intent but via feature patterns, not hard conditions.
        """
        if mf.get("leakage", 0):
            return "leakage_priority"
        if mf.get("baseline_alert", 0):
            return "baseline_priority"
        if mf.get("overfitting", 0) and float(mf.get("n_rows", 999)) < 500:
            return "overfit_small"
        if float(stack_rate) >= 0.60:
            return "stacking_wins"
        if float(mf.get("imbalance_ratio", 1.0)) > 4.0:
            return "severe_imbalance"
        if float(mf.get("cv_std", 0.0)) > 0.08:
            return "high_cv_variance"
        eff_cv = float(cv_mean) if cv_mean else float(mf.get("cv_mean", 0.0))
        if 0.0 < eff_cv < 0.60:
            return "low_cv_historical"
        return "default"

    def _fit(self, records: List[Dict]) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            log.warning("[RecommendationModel] sklearn unavailable — skipping fit.")
            return

        X: List[List[float]] = []
        y: List[str]          = []
        for r in records:
            mf = r.get("meta_features", {})
            if not mf:
                continue
            label = self._derive_label(
                mf,
                stack_rate = float(r.get("stack_rate",   0.0)),
                cv_mean    = float(r.get("cv_mean")  or
                                   mf.get("cv_mean", 0.0)),
            )
            X.append(_meta_feature_vector(mf))
            y.append(label)

        if len(X) < 5 or len(set(y)) < 2:
            log.debug(
                "[RecommendationModel] Insufficient label variety — skipping fit."
            )
            return

        clf = GradientBoostingClassifier(
            n_estimators=60, max_depth=3,
            learning_rate=0.1, random_state=42,
        )
        clf.fit(X, y)
        self._clf = clf
        log.info(
            f"[RecommendationModel] Fitted — "
            f"{len(X)} records · {len(set(y))} categories."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SUGGESTION PREDICTOR  (replaces hardcoded names in cold-start insight)
# ─────────────────────────────────────────────────────────────────────────────

class ModelSuggestionPredictor:
    """
    GradientBoosting classifier that learns which model family wins most
    often for different dataset profiles.

    Replaces the hardcoded LightGBM/RandomForest vs LogisticRegression/Ridge
    lists in _build_heuristic_insight().  Once trained, it predicts model
    family probabilities and maps them to representative model names.

    Falls back to the hardcoded defaults until MIN_SAMPLES records exist.
    """

    MIN_SAMPLES   = 10
    RETRAIN_EVERY = 8

    # Representative model names per family (for display in insights)
    _FAMILY_DISPLAY: Dict[str, List[str]] = {
        "boosting": ["LightGBM",          "XGBClassifier", "CatBoost"],
        "tree":     ["RandomForest",       "ExtraTrees",    "GradientBoosting"],
        "linear":   ["LogisticRegression", "Ridge",         "ElasticNet"],
        "bayesian": ["GaussianNB",         "BernoulliNB"],
    }

    # Keywords for mapping best_model string → family label
    _FAMILY_KEYWORDS: Dict[str, List[str]] = {
        "boosting": ["xgb", "lgbm", "lightgbm", "catboost"],
        "tree":     ["randomforest", "extratrees", "gradientboosting",
                     "histgradient", "decisiontree", "adaboost", "bagging"],
        "linear":   ["logistic", "linear", "ridge", "lasso",
                     "elastic", "sgd", "linearsvc"],
        "bayesian": ["gaussiannb", "bernoullinb", "multinomialnb", "qda"],
    }

    def __init__(self) -> None:
        self._clf          = None
        self._next_retrain = self.MIN_SAMPLES

    # ── Public ────────────────────────────────────────────────────────────────

    def maybe_fit(self, records: List[Dict]) -> None:
        """Re-fit if enough labelled records are available."""
        if len(records) >= self._next_retrain:
            self._fit(records)
            self._next_retrain = len(records) + self.RETRAIN_EVERY

    def predict_top_models(self, meta_features: Dict) -> List[Dict]:
        """
        Return a ranked list of {model, win_rate, wins} dicts for cold-start
        display.  Uses predicted family probabilities when the model is fitted;
        falls back to default suggestions otherwise.
        """
        if self._clf is not None:
            try:
                vec   = _meta_feature_vector(meta_features)
                proba = self._clf.predict_proba([vec])[0]
                family_probs = {
                    str(cls): float(p)
                    for cls, p in zip(self._clf.classes_, proba)
                }
                ranked = sorted(
                    family_probs.items(), key=lambda kv: kv[1], reverse=True
                )
                result = []
                for family, prob in ranked[:2]:
                    models = self._FAMILY_DISPLAY.get(family, ["UnknownModel"])
                    result.append({
                        "model":    models[0],
                        "win_rate": round(prob, 3),
                        "wins":     0,   # cold start — no wins tracked yet
                    })
                if result:
                    log.info(
                        f"[ModelSuggestionPredictor] → "
                        f"{[r['model'] for r in result]}"
                    )
                    return result
            except Exception as exc:
                log.warning(f"[ModelSuggestionPredictor] predict failed: {exc}")

        # Safety fallback — rule active only before model is trained
        return self._default_suggestions(meta_features)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _model_to_family(self, model_name: str) -> Optional[str]:
        """Map a model name string to one of the four family labels."""
        name_lc = model_name.lower().replace("_", "").replace(" ", "")
        for family, keywords in self._FAMILY_KEYWORDS.items():
            if any(kw in name_lc for kw in keywords):
                return family
        return None

    def _default_suggestions(self, meta_features: Dict) -> List[Dict]:
        """
        Minimal safety fallback — used only before model is trained.
        Retains the original hardcoded logic as a last resort.
        """
        n_rows = float(meta_features.get("n_rows", 0))
        if n_rows >= 1000:
            return [
                {"model": "LightGBM",     "win_rate": None, "wins": 0},
                {"model": "RandomForest", "win_rate": None, "wins": 0},
            ]
        return [
            {"model": "LogisticRegression", "win_rate": None, "wins": 0},
            {"model": "Ridge",              "win_rate": None, "wins": 0},
        ]

    def _fit(self, records: List[Dict]) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            log.warning("[ModelSuggestionPredictor] sklearn unavailable — skipping fit.")
            return

        X: List[List[float]] = []
        y: List[str]          = []
        for r in records:
            mf         = r.get("meta_features", {})
            model_name = r.get("best_model", "")
            if not mf or not model_name:
                continue
            family = self._model_to_family(model_name)
            if family is None:
                continue
            X.append(_meta_feature_vector(mf))
            y.append(family)

        if len(X) < 5 or len(set(y)) < 2:
            log.debug(
                "[ModelSuggestionPredictor] Insufficient label variety — skipping fit."
            )
            return

        clf = GradientBoostingClassifier(
            n_estimators=50, max_depth=3,
            learning_rate=0.1, random_state=42,
        )
        clf.fit(X, y)
        self._clf = clf
        log.info(
            f"[ModelSuggestionPredictor] Fitted — "
            f"{len(X)} records · {len(set(y))} families."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE RISK THRESHOLD  (replaces hardcoded 0.40 in risk flag logic)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRiskThreshold:
    """
    Learns per-flag alert thresholds from the distribution of observed
    risk rates across all profiles in the MetaTable.

    Replaces the hardcoded 0.40 threshold in _build_insight_from_table().

    Algorithm
    ─────────
    For each risk flag, collect the observed rate (count / n_runs) across
    all profiles.  Set the alert threshold = 75th percentile of those rates,
    clamped to [0.20, 0.70].

    A flag rate only triggers a risk message when it EXCEEDS this learned
    threshold, so the threshold self-calibrates to the system's own base rates
    — high base-rate flags require a higher bar to signal as "unusual".

    Falls back to DEFAULT=0.40 until MIN_PROFILES profiles are seen.
    """

    MIN_PROFILES = 5
    DEFAULT      = 0.40

    _FLAGS = ["overfitting", "leakage", "baseline_alert"]

    def __init__(self) -> None:
        self._thresholds: Dict[str, float] = {
            f: self.DEFAULT for f in self._FLAGS
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, table: MetaTable) -> None:
        """Recompute thresholds from current MetaTable state."""
        if table.n_profiles() < self.MIN_PROFILES:
            return   # safety: keep defaults until enough profiles exist

        flag_rates: Dict[str, List[float]] = {f: [] for f in self._FLAGS}
        for entry in table._table.values():
            n = entry.get("n_runs", 0)
            if n == 0:
                continue
            risk_counts = entry.get("risk_counts", {})
            for flag in self._FLAGS:
                rate = risk_counts.get(flag, 0) / n
                flag_rates[flag].append(rate)

        for flag, rates in flag_rates.items():
            if len(rates) >= 3:
                # 75th percentile → adaptive alert threshold
                p75 = float(np.percentile(rates, 75))
                # Clamp to [0.20, 0.70] — safety bounds
                self._thresholds[flag] = max(0.20, min(0.70, p75))

        log.debug(
            f"[AdaptiveRiskThreshold] Updated → {self._thresholds}"
        )

    def threshold(self, flag: str) -> float:
        """Return the learned threshold for a given risk flag."""
        return self._thresholds.get(flag, self.DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL INSTANCES  (shared across module-level helpers)
# ─────────────────────────────────────────────────────────────────────────────

_global_rec_model      = RecommendationModel()
_global_model_suggester= ModelSuggestionPredictor()
_global_risk_threshold = AdaptiveRiskThreshold()


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CONFIDENCE SCORING  (replaces hardcoded n>=10 / n>=3 thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_confidence(n: int, scores: List[float]) -> str:
    """
    v2.0 — Compute insight confidence as a weighted combination of:
      - count_score  : saturates at n=20 (60% weight)
      - stability    : 1 - normalised score spread (40% weight)

    Replaces the hardcoded "High if n>=10 else Medium if n>=3 else Low"
    so a profile with 12 highly volatile scores is not rated "High".

    Thresholds: quality >= 0.65 → High, >= 0.30 → Medium, else Low.
    """
    if n == 0:
        return "Low"

    count_score = min(n, 20) / 20.0   # saturates at 20 runs

    if len(scores) >= 2:
        spread = statistics.stdev(scores)
        # 0.15 stdev = "noisy" — maps to stability=0
        stability = max(0.0, 1.0 - spread / 0.15)
    else:
        stability = 0.50   # single sample — uncertain

    quality = count_score * 0.60 + stability * 0.40

    if quality >= 0.65:
        return "High"
    if quality >= 0.30:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT BUILDER  (converts table entry → human-readable insight dict)
# ─────────────────────────────────────────────────────────────────────────────

def _build_insight_from_table(
    entry:         Dict,
    meta_features: Dict,
    profile:       str,
) -> Dict:
    """
    Build a structured insight dict from a MetaTable entry.

    v2.0 changes:
      • Risk flag threshold → _global_risk_threshold.threshold(flag)
        instead of hardcoded 0.40
      • Confidence → _adaptive_confidence(n, scores) instead of
        hardcoded n>=10 / n>=3 thresholds
    """
    n          = entry["n_runs"]
    scores     = entry["cv_scores"]
    model_wins = entry.get("model_wins", {})
    risk_counts= entry.get("risk_counts", {})
    stack_wins = entry.get("stacking_wins", 0)

    # ── Expected CV range ─────────────────────────────────────────────────────
    if len(scores) >= 3:
        lo = round(float(np.percentile(scores, 10)), 4)
        hi = round(float(np.percentile(scores, 90)), 4)
        mu = round(float(np.mean(scores)), 4)
    elif scores:
        lo = mu = hi = round(float(np.mean(scores)), 4)
    else:
        lo = mu = hi = None

    # ── Top model predictions ─────────────────────────────────────────────────
    total_wins = sum(model_wins.values()) or 1
    top_models = sorted(model_wins.items(),
                        key=lambda kv: kv[1], reverse=True)[:5]
    top_model_predictions = [
        {"model": m, "win_rate": round(c / total_wins, 3), "wins": c}
        for m, c in top_models
    ]

    # ── Risk flags — adaptive threshold ──────────────────────────────────────
    # v2.0: threshold per flag is learned from rate distributions, not 0.40
    risk_flags = []
    for flag, count in risk_counts.items():
        rate      = count / n
        threshold = _global_risk_threshold.threshold(flag)
        if rate >= threshold:
            risk_flags.append(
                f"{flag.replace('_', ' ').title()} detected in "
                f"{rate*100:.0f}% of similar runs "
                f"(threshold: {threshold*100:.0f}%) — review carefully."
            )

    # Current-run flags
    if meta_features.get("overfitting"):
        risk_flags.append(
            "Current run: overfitting detected (train/CV gap > 10%).")
    if meta_features.get("leakage"):
        risk_flags.append(
            "Current run: data leakage detected; removed features may carry "
            "target signal.")
    if meta_features.get("baseline_alert"):
        risk_flags.append(
            "Current run: model barely beats majority-class baseline.")

    # ── Stacking insight ──────────────────────────────────────────────────────
    stack_rate = stack_wins / n if n > 0 else 0.0
    if stack_rate >= 0.50:
        risk_flags.append(
            f"Stacking outperformed single models in {stack_rate*100:.0f}% of "
            f"similar runs — consider BOOST_ENSEMBLE action."
        )

    # ── Recommendation ────────────────────────────────────────────────────────
    recommendation = _make_recommendation(
        meta_features, top_models, stack_rate, scores
    )

    # ── Insight confidence — adaptive ─────────────────────────────────────────
    # v2.0: considers both sample count and score stability
    conf = _adaptive_confidence(n, scores)

    return {
        "profile":               profile,
        "profile_match_count":   n,
        "expected_cv_range":     [lo, hi],
        "expected_cv_mean":      mu,
        "top_model_predictions": top_model_predictions,
        "stacking_win_rate":     round(stack_rate, 3),
        "risk_flags":            risk_flags,
        "recommendation":        recommendation,
        "confidence":            conf,
        "source":                "meta_table",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION STRING HELPERS  (used by _make_recommendation)
# ─────────────────────────────────────────────────────────────────────────────

def _category_to_rec_string(
    category:      str,
    meta_features: Dict,
    top_models:    List[Tuple],
    stack_rate:    float,
    scores:        List[float],
) -> Optional[str]:
    """
    Map a predicted recommendation category to the final recommendation string.

    Returns None for "model_preference" when top_models is empty (so the
    caller can fall back to the rule chain) and for unrecognised categories.
    """
    if category == "leakage_priority":
        return ("Data leakage was detected — audit feature-target correlations "
                "before trusting model performance.")
    if category == "baseline_priority":
        return ("Model barely beats the majority-class baseline — "
                "consider richer features or a larger dataset.")
    if category == "overfit_small":
        return ("Overfitting on a small dataset — increase regularisation "
                "or collect more data.")
    if category == "stacking_wins":
        return ("Stacking consistently outperforms single models on this "
                "profile — use BOOST_ENSEMBLE action.")
    if category == "severe_imbalance":
        return ("Severe class imbalance detected — verify "
                "class_weight='balanced' and consider SMOTE or targeted "
                "oversampling.")
    if category == "high_cv_variance":
        return ("High CV variance — use INCREASE_CV_FOLDS for more stable "
                "estimates.")
    if category == "model_preference":
        if top_models:
            leader = top_models[0][0]
            return (f"'{leader}' wins most often on this dataset profile — "
                    f"prioritise it or use PRIORITISE_BOOSTING / "
                    f"PRIORITISE_LINEAR.")
        return None   # fall through — top_models unavailable
    if category == "low_cv_historical":
        return ("Historical CV scores for this profile are low — "
                "consider feature engineering or additional data collection.")
    if category == "default":
        return "No specific concerns — default pipeline settings are appropriate."
    return None


def _make_recommendation(
    meta_features: Dict,
    top_models:    List[Tuple],
    stack_rate:    float,
    scores:        List[float],
) -> str:
    """
    Single-sentence actionable recommendation.

    v2.0 — model-first: tries RecommendationModel classifier first.
    Maps predicted category to recommendation string via
    _category_to_rec_string().  Falls back to the minimal rule chain only
    when the model is absent, low-confidence, or returns None for a
    category that requires additional context (e.g. top_models empty).

    Rules retained below are a SAFETY NET, not primary logic.
    """
    # ── Primary: learned classifier ───────────────────────────────────────────
    category = _global_rec_model.predict(meta_features)
    if category is not None:
        rec = _category_to_rec_string(
            category, meta_features, top_models, stack_rate, scores
        )
        if rec:
            return rec

    # ── Safety fallback — rule chain (active only when model absent/low-conf) ─
    n_rows = meta_features.get("n_rows", 0)
    imb    = meta_features.get("imbalance_ratio", 1.0)
    cv_std = meta_features.get("cv_std", 0.0)

    if meta_features.get("leakage"):
        return ("Data leakage was detected — audit feature-target correlations "
                "before trusting model performance.")
    if meta_features.get("baseline_alert"):
        return ("Model barely beats the majority-class baseline — "
                "consider richer features or a larger dataset.")
    if meta_features.get("overfitting") and n_rows < 500:
        return ("Overfitting on a small dataset — increase regularisation "
                "or collect more data.")
    if stack_rate >= 0.60:
        return ("Stacking consistently outperforms single models on this "
                "profile — use BOOST_ENSEMBLE action.")
    if imb and imb > 4.0:
        return ("Severe class imbalance detected — verify "
                "class_weight='balanced' and consider SMOTE or targeted "
                "oversampling.")
    if cv_std and cv_std > 0.08:
        return ("High CV variance — use INCREASE_CV_FOLDS for more stable "
                "estimates.")
    if top_models:
        leader = top_models[0][0]
        return (f"'{leader}' wins most often on this dataset profile — "
                f"prioritise it or use PRIORITISE_BOOSTING / PRIORITISE_LINEAR.")
    if scores and float(np.mean(scores)) < 0.60:
        return ("Historical CV scores for this profile are low — "
                "consider feature engineering or additional data collection.")
    return "No specific concerns — default pipeline settings are appropriate."


def _build_heuristic_insight(
    meta_features: Dict, profile: str
) -> Dict:
    """
    v2.0 — Insight for unseen profiles (cold start).

    Uses ModelSuggestionPredictor for top-model recommendations when
    trained; falls back to the minimal hardcoded defaults otherwise.

    Risk flags remain rule-based here because they operate on raw
    feature values (not historical rates) — this is system safety logic,
    not a decision to be learned.
    """
    n_rows  = meta_features.get("n_rows", 0)
    n_cols  = meta_features.get("n_cols", 0)
    missing = meta_features.get("missing_pct", 0)
    imb     = meta_features.get("imbalance_ratio", 1.0)

    # Safety rules — raw feature guards, kept intentionally
    risk_flags = []
    if n_rows < 200:
        risk_flags.append(
            "Very small dataset (<200 rows) — results may be unstable.")
    if missing > 20:
        risk_flags.append(
            f"High missing rate ({missing:.1f}%) — imputation may introduce bias.")
    if imb and imb > 4.0:
        risk_flags.append(
            f"Severe class imbalance (ratio={imb:.1f}) — monitor minority-class recall.")
    if n_cols > n_rows:
        risk_flags.append(
            "More features than rows — strong regularisation recommended.")

    # v2.0 — model-based top model suggestions (falls back to defaults)
    top_model_predictions = _global_model_suggester.predict_top_models(
        meta_features
    )

    recommendation = _make_recommendation(meta_features, [], 0.0, [])

    return {
        "profile":               profile,
        "profile_match_count":   0,
        "expected_cv_range":     [None, None],
        "expected_cv_mean":      None,
        "top_model_predictions": top_model_predictions,
        "stacking_win_rate":     None,
        "risk_flags":            risk_flags,
        "recommendation":        recommendation,
        "confidence":            "Low",
        "source":                "heuristic_cold_start",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN META-REGRESSOR  (Phase 2 — kicks in after MIN_RECORDS runs)
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_META_KEYS = [
    "n_rows", "n_cols", "missing_pct", "n_classes", "imbalance_ratio",
    "target_skew", "cv_std", "overfitting", "leakage", "baseline_alert",
    "orig_features", "final_features", "fs_ratio", "pca_applied",
    "lda_applied", "tier", "n_cv_folds",
]


def _meta_feature_vector(mf: Dict) -> List[float]:
    """Extract a fixed-length numeric vector from a meta-feature dict."""
    return [float(mf.get(k, 0.0) or 0.0) for k in _NUMERIC_META_KEYS]


def _fit_sklearn_meta_model(records: List[Dict]):
    """
    Fit a GradientBoostingRegressor to predict CV score from meta-features.

    records: list of dicts each containing meta_features + cv_mean
    Returns fitted sklearn model or None on failure.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    valid = [(r["meta_features"], r["cv_mean"])
             for r in records
             if "meta_features" in r and r.get("cv_mean") is not None
             and not math.isnan(float(r["cv_mean"]))]
    if len(valid) < MIN_RECORDS_FOR_FIT:
        return None

    X = np.array([_meta_feature_vector(mf) for mf, _ in valid])
    y = np.array([float(cv) for _, cv in valid])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42,
        )),
    ])
    try:
        pipe.fit(X, y)
        log.info(f"[MetaModel] Fitted sklearn regressor on {len(valid)} records.")
        return pipe
    except Exception as exc:
        log.warning(f"[MetaModel] Sklearn fit failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# META MODEL  (main class)
# ─────────────────────────────────────────────────────────────────────────────

class MetaModel:
    """
    Two-phase cross-run pattern recogniser.

    Usage
    ─────
    mm = MetaModel.load_or_create()
    insight = mm.predict(pipeline_result)
    mm.record(pipeline_result)     # call AFTER each run to accumulate data
    mm.maybe_refit()               # call periodically; fits sklearn if enough data
    mm.save()

    v2.0 — wires RecommendationModel, ModelSuggestionPredictor, and
    AdaptiveRiskThreshold into the record() / load_or_create() lifecycle
    so all three learned models update incrementally alongside the main
    Phase 2 regressor.
    """

    VERSION = "2.0"

    def __init__(
        self,
        table:             MetaTable      | None = None,
        sklearn_model:                      Any  = None,
        total_records:     int                   = 0,
        last_fitted:       Optional[str]         = None,
        records_since_fit: int                   = 0,
    ) -> None:
        self.table             = table or MetaTable()
        self.sklearn_model     = sklearn_model     # None until Phase 2
        self.total_records     = total_records
        self.last_fitted       = last_fitted
        self.records_since_fit = records_since_fit
        self._pending_records: List[Dict] = []     # in-memory buffer for refit

        # v2.0 — references to global learned sub-models
        self._rec_model       = _global_rec_model
        self._model_suggester = _global_model_suggester
        self._risk_threshold  = _global_risk_threshold

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls) -> "MetaModel":
        """Load from meta_state/ if files exist, else fresh instance."""
        # Load MetaTable
        table_data = _load_json(_META_TABLE_PATH)
        table = MetaTable.from_dict(table_data) if table_data else MetaTable()

        # Load sklearn model
        sklearn_model = None
        if _META_MODEL_PATH.exists():
            try:
                sklearn_model = joblib.load(_META_MODEL_PATH)
                log.info("[MetaModel] Loaded sklearn regressor from disk.")
            except Exception as exc:
                log.warning(f"[MetaModel] Could not load sklearn model: {exc}")

        # Load metadata
        meta = _load_json(_META_META_PATH)
        inst = cls(
            table             = table,
            sklearn_model     = sklearn_model,
            total_records     = int(meta.get("total_records",     0)),
            last_fitted       = meta.get("last_fitted"),
            records_since_fit = int(meta.get("records_since_fit", 0)),
        )

        # v2.0 — warm up learned sub-models from any persisted pending records
        # and update adaptive threshold from the loaded table
        if inst._pending_records:
            inst._rec_model.maybe_fit(inst._pending_records)
            inst._model_suggester.maybe_fit(inst._pending_records)
        inst._risk_threshold.update(inst.table)

        log.info(
            f"[MetaModel] Loaded — records={inst.total_records}, "
            f"profiles={table.n_profiles()}, "
            f"phase={'2 (sklearn)' if sklearn_model else '1 (table)'}"
        )
        return inst

    # ── Record a completed run ────────────────────────────────────────────────

    def record(self, pipeline_result: Dict) -> None:
        """
        Ingest a completed pipeline result into the MetaTable and pending buffer.
        Call this after every successful run.

        v2.0 — also triggers incremental re-fitting of RecommendationModel,
        ModelSuggestionPredictor, and AdaptiveRiskThreshold.
        """
        mf         = extract_meta_features(pipeline_result)
        profile    = dataset_profile(mf)
        best_model = (pipeline_result.get("best_model_name")
                      or pipeline_result.get("best_model", "unknown"))
        stack_name = (pipeline_result.get("stacking_model", "N/A") or "N/A")
        stacking_won = (stack_name != "N/A" and stack_name == best_model)

        self.table.record(profile, mf, best_model, stacking_won)

        # Buffer for potential sklearn refit — v2.0: include extra fields
        self._pending_records.append({
            "meta_features": mf,
            "cv_mean":       mf.get("cv_mean"),
            "profile":       profile,
            "best_model":    best_model,        # for ModelSuggestionPredictor
            "stack_rate":    float(stacking_won),  # for RecommendationModel
        })
        self.total_records     += 1
        self.records_since_fit += 1

        # v2.0 — incremental re-fit of learned sub-models
        self._rec_model.maybe_fit(self._pending_records)
        self._model_suggester.maybe_fit(self._pending_records)
        self._risk_threshold.update(self.table)   # O(profiles) — fast

        log.info(
            f"[MetaModel] Recorded run — profile={profile!r}, "
            f"model={best_model}, total={self.total_records}"
        )

    # ── Predict insight ───────────────────────────────────────────────────────

    def predict(self, pipeline_result: Dict) -> Dict:
        """
        Generate an insight dict for the given pipeline result.

        If a sklearn model is fitted (Phase 2) it augments the table
        insight with a predicted CV score.

        Returns a structured insight dict — always succeeds (falls back
        to heuristics if the table has no data for this profile).
        """
        mf      = extract_meta_features(pipeline_result)
        profile = dataset_profile(mf)
        entry   = self.table.query(profile)

        if entry and entry["n_runs"] > 0:
            insight = _build_insight_from_table(entry, mf, profile)
        else:
            insight = _build_heuristic_insight(mf, profile)

        # Phase 2 augmentation
        if self.sklearn_model is not None:
            try:
                vec = np.array(_meta_feature_vector(mf)).reshape(1, -1)
                predicted_cv = float(self.sklearn_model.predict(vec)[0])
                predicted_cv = round(max(0.0, min(1.0, predicted_cv)), 4)
                insight["sklearn_predicted_cv"] = predicted_cv
                insight["source"] = (
                    "meta_table+sklearn"
                    if entry and entry["n_runs"] > 0
                    else "sklearn_only"
                )
                log.info(
                    f"[MetaModel] sklearn predicted CV = {predicted_cv:.4f}"
                )
            except Exception as exc:
                log.warning(f"[MetaModel] sklearn prediction failed: {exc}")

        insight["total_records_seen"] = self.total_records
        log.info(
            f"[MetaModel] Insight — profile={profile!r}, "
            f"matches={insight['profile_match_count']}, "
            f"conf={insight['confidence']}"
        )
        return insight

    # ── Refit sklearn model ───────────────────────────────────────────────────

    def maybe_refit(self, force: bool = False) -> bool:
        """
        Refit the sklearn meta-regressor if enough new records have accumulated.

        Triggers when:
          • total_records >= MIN_RECORDS_FOR_FIT, AND
          • records_since_fit >= 5 (at least 5 new runs since last fit), OR
          • force=True

        Returns True if a new model was fitted.
        """
        enough_total = self.total_records >= MIN_RECORDS_FOR_FIT
        enough_new   = self.records_since_fit >= 5
        if not force and not (enough_total and enough_new):
            return False

        log.info(
            f"[MetaModel] Fitting sklearn regressor "
            f"(total={self.total_records}, new_since_fit={self.records_since_fit})"
        )

        # Reconstruct full record list from MetaTable (safe fallback)
        # plus any in-memory pending records
        all_records: List[Dict] = list(self._pending_records)

        new_model = _fit_sklearn_meta_model(all_records)
        if new_model is not None:
            self.sklearn_model     = new_model
            self.last_fitted       = _ts()
            self.records_since_fit = 0
            log.info("[MetaModel] sklearn model updated.")
            return True
        return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Flush all state to meta_state/."""
        _save_json(_META_TABLE_PATH, self.table.to_dict())
        if self.sklearn_model is not None:
            try:
                joblib.dump(self.sklearn_model, _META_MODEL_PATH)
            except Exception as exc:
                log.warning(f"[MetaModel] Could not save sklearn model: {exc}")
        _save_json(_META_META_PATH, {
            "version":          self.VERSION,
            "total_records":    self.total_records,
            "records_since_fit":self.records_since_fit,
            "last_fitted":      self.last_fitted,
            "phase":            2 if self.sklearn_model else 1,
            "n_profiles":       self.table.n_profiles(),
            "last_saved":       _ts(),
        })
        log.info(
            f"[MetaModel] Saved — records={self.total_records}, "
            f"profiles={self.table.n_profiles()}"
        )

    def full_report(self) -> Dict:
        """Full snapshot for /agent_status endpoint."""
        return {
            "version":          self.VERSION,
            "total_records":    self.total_records,
            "n_profiles":       self.table.n_profiles(),
            "phase":            2 if self.sklearn_model else 1,
            "last_fitted":      self.last_fitted,
            "records_since_fit":self.records_since_fit,
            "min_for_phase2":   MIN_RECORDS_FOR_FIT,
            # v2.0 — expose learned sub-model states
            "rec_model_ready":          self._rec_model._clf is not None,
            "model_suggester_ready":    self._model_suggester._clf is not None,
            "learned_risk_thresholds":  self._risk_threshold._thresholds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    except Exception as exc:
        log.warning(f"[MetaModel] Could not load {path}: {exc}")
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
    except Exception as exc:
        log.warning(f"[MetaModel] Could not save {path}: {exc}")


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    print("\n── meta_model.py self-tests (v2.0) ──")
    import tempfile

    # ── 1. dataset_profile bucketing ─────────────────────────────────────────
    p = dataset_profile({
        "n_rows": 5000, "n_cols": 30, "overall_missing_pct": 3.0,
        "imbalance_ratio": 2.5, "problem_type": "classification",
    })
    assert "classification" in p
    assert "rows:small"     in p
    assert "feat:medium"    in p
    assert "mis:clean"      in p
    assert "imb:moderate"   in p
    print(f"✓ dataset_profile OK  →  {p}")

    # ── 2. extract_meta_features ──────────────────────────────────────────────
    mock_result = {
        "problem_type": "classification",
        "best_model_name": "LightGBM",
        "stacking_model": "N/A",
        "scale_tier": 1,
        "dataset_diagnostics": {
            "n_rows": 1200, "n_cols": 22,
            "overall_missing_pct": 5.5,
            "imbalance_ratio": 3.1,
            "n_classes": 3,
        },
        "performance": {
            "cv_score_mean":  0.83,
            "cv_score_std":   0.04,
            "confidence_score": 0.79,
            "overfitting":    False,
            "leakage_detected": False,
            "n_cv_folds":     3,
            "feature_selection": {
                "original_features": 22,
                "final_features":    15,
                "pca_applied":       False,
                "lda_applied":       False,
            },
        },
        "baseline_alert": {"triggered": False},
    }
    mf = extract_meta_features(mock_result)
    assert mf["n_rows"]      == 1200
    assert mf["cv_mean"]     == 0.83
    assert mf["overfitting"] == 0
    assert mf["fs_ratio"]    == round(15 / 22, 4)
    print(f"✓ extract_meta_features OK  keys={list(mf.keys())[:5]}…")

    # ── 3. MetaTable record + query ───────────────────────────────────────────
    mt   = MetaTable()
    prof = dataset_profile(mf)
    for score in [0.80, 0.82, 0.85, 0.79, 0.88]:
        mf2 = dict(mf); mf2["cv_mean"] = score
        mt.record(prof, mf2, "LightGBM", stacking_won=False)
    mt.record(prof, dict(mf), "RandomForest", stacking_won=True)
    entry = mt.query(prof)
    assert entry["n_runs"]                     == 6
    assert entry["model_wins"]["LightGBM"]     == 5
    assert entry["model_wins"]["RandomForest"] == 1
    assert entry["stacking_wins"]              == 1
    print(f"✓ MetaTable record/query OK  n_runs={entry['n_runs']}")

    # ── 4. MetaTable serialisation round-trip ─────────────────────────────────
    d   = mt.to_dict()
    mt2 = MetaTable.from_dict(d)
    assert mt2.query(prof)["n_runs"] == 6
    print("✓ MetaTable round-trip OK")

    # ── 5. Insight from table ─────────────────────────────────────────────────
    insight = _build_insight_from_table(entry, mf, prof)
    assert insight["profile_match_count"]                == 6
    assert insight["top_model_predictions"][0]["model"]  == "LightGBM"
    assert insight["expected_cv_range"][0] is not None
    # v2.0: confidence is now adaptive — just check it returns a valid string
    assert insight["confidence"] in ("High", "Medium", "Low")
    print(f"✓ _build_insight_from_table OK  conf={insight['confidence']}")

    # ── 6. Adaptive confidence scoring ────────────────────────────────────────
    # High count + stable scores → High
    conf_h = _adaptive_confidence(20, [0.80, 0.81, 0.82, 0.80, 0.81] * 4)
    # Low count → Low or Medium
    conf_l = _adaptive_confidence(1, [0.80])
    # High count + volatile scores → lower confidence
    conf_v = _adaptive_confidence(20, [0.50, 0.95, 0.60, 0.90, 0.55] * 4)
    assert conf_h in ("High", "Medium")
    assert conf_l in ("Low", "Medium")
    assert conf_v in ("Medium", "Low")
    assert conf_h != conf_v, "Stable scores should score higher than volatile"
    print(f"✓ _adaptive_confidence OK  stable={conf_h}, volatile={conf_v}, low={conf_l}")

    # ── 7. Heuristic cold-start insight ──────────────────────────────────────
    hi = _build_heuristic_insight(
        {"n_rows": 100, "problem_type": "classification",
         "imbalance_ratio": 6.0, "missing_pct": 25.0,
         "n_cols": 200, "cv_std": 0.0}, "test_profile"
    )
    assert hi["confidence"]      == "Low"
    assert hi["source"]          == "heuristic_cold_start"
    assert len(hi["risk_flags"]) >= 2    # small dataset + high missing
    assert len(hi["top_model_predictions"]) >= 1
    print(f"✓ Heuristic insight OK  risks={len(hi['risk_flags'])}, "
          f"top_model={hi['top_model_predictions'][0]['model']}")

    # ── 8. AdaptiveRiskThreshold — defaults before enough profiles ────────────
    art = AdaptiveRiskThreshold()
    assert art.threshold("overfitting")    == 0.40
    assert art.threshold("leakage")        == 0.40
    assert art.threshold("baseline_alert") == 0.40
    print("✓ AdaptiveRiskThreshold defaults OK")

    # ── 9. AdaptiveRiskThreshold — adapts after enough profiles ───────────────
    mt3 = MetaTable()
    for i in range(8):
        prof3 = f"classification|rows:small|feat:medium|mis:clean|imb:balanced_{i}"
        # Inject high overfitting rates to drive threshold up
        mt3._table[prof3] = {
            "n_runs": 10,
            "cv_scores": [0.75] * 10,
            "model_wins": {"LightGBM": 10},
            "risk_counts": {"overfitting": 8, "leakage": 1},   # 80% overfit rate
            "stacking_wins": 2,
        }
    art2 = AdaptiveRiskThreshold()
    art2.update(mt3)
    # With 80% overfit rates, threshold should be pushed above 0.40
    assert art2.threshold("overfitting") > 0.40, (
        f"Expected threshold > 0.40, got {art2.threshold('overfitting')}")
    print(f"✓ AdaptiveRiskThreshold update OK  "
          f"overfit_threshold={art2.threshold('overfitting'):.3f}")

    # ── 10. RecommendationModel — returns None before training ────────────────
    rm = RecommendationModel()
    pred = rm.predict(mf)
    assert pred is None, "Expected None before training"
    print("✓ RecommendationModel returns None before training OK")

    # ── 11. ModelSuggestionPredictor — falls back before training ─────────────
    msp = ModelSuggestionPredictor()
    sug = msp.predict_top_models({"n_rows": 5000})
    assert len(sug) >= 1
    assert sug[0]["model"] in ("LightGBM", "RandomForest",
                               "LogisticRegression", "Ridge")
    print(f"✓ ModelSuggestionPredictor fallback OK  → {sug[0]['model']}")

    # ── 12. _make_recommendation — falls back to rules before model ───────────
    rec = _make_recommendation(
        {"overfitting": 1, "n_rows": 150, "leakage": 0, "baseline_alert": 0,
         "imbalance_ratio": 1.1, "cv_std": 0.02},
        [], 0.0, []
    )
    assert "overfit" in rec.lower() or "small" in rec.lower()
    print(f"✓ _make_recommendation fallback OK  →  '{rec[:60]}…'")

    # ── 13. MetaModel full lifecycle (temp dir) ───────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        global _META_TABLE_PATH, _META_MODEL_PATH, _META_META_PATH, _META_STATE_DIR
        _orig = (_META_TABLE_PATH, _META_MODEL_PATH, _META_META_PATH, _META_STATE_DIR)
        _META_STATE_DIR  = Path(tmp)
        _META_TABLE_PATH = _META_STATE_DIR / "meta_table.json"
        _META_MODEL_PATH = _META_STATE_DIR / "meta_model.pkl"
        _META_META_PATH  = _META_STATE_DIR / "meta_meta.json"

        mm = MetaModel.load_or_create()
        assert mm.total_records == 0

        # Record enough runs to trigger refit
        for i in range(25):
            r = dict(mock_result)
            r["performance"] = dict(mock_result["performance"])
            r["performance"]["cv_score_mean"] = 0.75 + i * 0.005
            mm.record(r)

        assert mm.total_records     == 25
        assert mm.records_since_fit == 25

        refitted = mm.maybe_refit()
        assert refitted is True, "Expected refit after 25 records"
        assert mm.sklearn_model is not None
        print("✓ MetaModel.maybe_refit triggered sklearn fit OK")

        # v2.0 — check sub-models were triggered
        report = mm.full_report()
        assert "rec_model_ready"       in report
        assert "model_suggester_ready" in report
        assert "learned_risk_thresholds" in report
        print(f"✓ MetaModel.full_report v2.0 fields OK  "
              f"rec_ready={report['rec_model_ready']}, "
              f"suggester_ready={report['model_suggester_ready']}")

        # predict
        insight2 = mm.predict(mock_result)
        assert "sklearn_predicted_cv" in insight2
        assert 0.0 <= insight2["sklearn_predicted_cv"] <= 1.0
        print(f"✓ MetaModel.predict with sklearn OK  "
              f"predicted_cv={insight2['sklearn_predicted_cv']:.4f}")

        mm.save()
        mm3 = MetaModel.load_or_create()
        assert mm3.total_records == 25
        assert mm3.sklearn_model is not None
        print("✓ MetaModel save/load round-trip OK")

        _META_TABLE_PATH, _META_MODEL_PATH, _META_META_PATH, _META_STATE_DIR = _orig

    print("\n✓ All meta_model.py v2.0 self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()