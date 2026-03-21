"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   UNIVERSAL AUTOML PIPELINE  v5.5  —  FULL LEARNING AI SYSTEM              ║
║                                                                              ║
║  v5.5 AGENT INTEGRATION:                                                    ║
║    [+] run_automl_extended()  — new recommended entry point                 ║
║    [+] automl_integration.py  — modular pipeline (run_full_pipeline)        ║
║    [+] rl_agent.py            — cross-run Q-table agent                     ║
║    [+] meta_model.py          — cross-run pattern recognition               ║
║    [+] retrain_model.py       — drift detection + retrain trigger           ║
║    [+] agent_system.py        — top-level orchestrator                      ║
║                                                                              ║
║  v5.6 LEARNING UPGRADES:                                                    ║
║    [+] MetaPriorityScorer     — GBM replaces hardcoded priority rules       ║
║    [+] SizePenaltyWeightModel — learned hybrid size penalty                 ║
║    [+] Hybrid calculate_confidence — learned weights for all penalties      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Inline path bootstrap (no external module needed) ──────────────────────
import sys as _sys, pathlib as _pl
_sd = str(_pl.Path(__file__).resolve().parent)
if _sd not in _sys.path: _sys.path.insert(0, _sd)
del _sd
# ────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# SECTION 1 — IMPORTS & LOGGING
# ==============================================================================
import re, math, uuid, random, warnings, logging, time
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib, shap
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, RidgeClassifier, Lasso, ElasticNet,
    BayesianRidge, SGDClassifier, SGDRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
log = logging.getLogger(__name__)

# ==============================================================================
# SECTION 2 — JSON-SAFE UTILITIES
# ==============================================================================

def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [convert_to_python(v) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return convert_to_python(obj.tolist())
    if isinstance(obj, pd.Series): return convert_to_python(obj.tolist())
    if isinstance(obj, pd.DataFrame): return convert_to_python(obj.to_dict(orient="records"))
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val); return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception: return None

# ==============================================================================
# SECTION 2.5 — DATASET DIAGNOSTICS
# ==============================================================================

def _log_dataset_diagnostics(df: "pd.DataFrame", target_column: str,
                              problem_type: str) -> dict:
    diag: dict = {}
    n_rows, n_cols = df.shape
    diag["n_rows"] = n_rows
    diag["n_cols"] = n_cols
    missing = (df.isnull().mean() * 100).round(2)
    top_miss = missing[missing > 0].sort_values(ascending=False).head(10).to_dict()
    diag["missing_pct_top10"]    = top_miss
    diag["overall_missing_pct"]  = round(float(df.isnull().values.mean() * 100), 2)
    if target_column in df.columns:
        y = df[target_column]
        if problem_type == "classification":
            vc = y.value_counts(normalize=True).round(4)
            diag["class_distribution"] = vc.to_dict()
            diag["n_classes"]          = int(y.nunique())
            maj  = float(vc.max())
            min_ = float(vc.min())
            diag["imbalance_ratio"] = round(maj / max(min_, 1e-9), 2)
            log.info(
                f"[Diagnostics] Target '{target_column}': {diag['n_classes']} classes, "
                f"imbalance_ratio={diag['imbalance_ratio']:.2f}"
            )
        else:
            y_num = pd.to_numeric(y, errors="coerce").dropna()
            diag["target_mean"] = round(float(y_num.mean()), 4)
            diag["target_std"]  = round(float(y_num.std()), 4)
            diag["target_skew"] = round(float(y_num.skew()), 4)
            diag["target_min"]  = round(float(y_num.min()), 4)
            diag["target_max"]  = round(float(y_num.max()), 4)
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([target_column])
    if len(num_cols) > 0:
        skews = df[num_cols].skew().abs().sort_values(ascending=False).head(5)
        diag["most_skewed_features"] = skews.round(3).to_dict()
    log.info(
        f"[Diagnostics] Shape={n_rows}×{n_cols}, "
        f"overall_missing={diag['overall_missing_pct']}%"
    )
    return diag


# ==============================================================================
# SECTION 2.6 — LEARNED PENALTY WEIGHT MODELS (v5.6)
# ==============================================================================

class SizePenaltyWeightModel:
    """
    Ridge regression that learns how much the dataset-size penalty
    should actually reduce confidence — replacing the hardcoded 0.25 scalar.

    Insight
    ───────
    The original rule: size_pen = (1 - n/200) * 0.25
    The hybrid v5.6:  size_pen = (1 - n/200) * learned_weight

    learned_weight is trained from the ledger: for small-dataset runs,
    it observes how much the raw CV score over-estimated the true test
    performance — and calibrates the weight to reflect that gap.

    Falls back to 0.25 when fewer than MIN_SAMPLES records exist.
    """

    MIN_SAMPLES = 20
    RETRAIN_EVERY = 10
    _DEFAULT_WEIGHT = 0.25
    _CLAMP = (0.05, 0.50)   # safety bounds

    def __init__(self) -> None:
        self._weight: float = self._DEFAULT_WEIGHT
        self._next_retrain: int = self.MIN_SAMPLES

    @property
    def weight(self) -> float:
        return self._weight

    def maybe_fit(self, ledger_records: list) -> None:
        small = [r for r in ledger_records
                 if (r.get("dataset_diagnostics") or {}).get("n_rows", 9999) < 200]
        if len(small) >= self._next_retrain:
            self._fit(small)
            self._next_retrain = len(small) + self.RETRAIN_EVERY

    def _fit(self, records: list) -> None:
        try:
            from sklearn.linear_model import Ridge as _Ridge
        except ImportError:
            return
        X, y = [], []
        for r in records:
            n    = float((r.get("dataset_diagnostics") or {}).get("n_rows", 100))
            cv   = float((r.get("metrics") or r.get("performance") or {}).get("cv_score_mean", 0.5) or 0.5)
            test = float((r.get("metrics") or r.get("performance") or {}).get(
                "accuracy") or (r.get("metrics") or r.get("performance") or {}).get("R2") or cv)
            gap  = max(0.0, cv - test)      # how much CV over-estimated
            raw_pen = max(0.0, 1.0 - n / 200.0)
            if raw_pen < 1e-6:
                continue
            # target: what weight would have predicted the gap correctly?
            target_w = gap / raw_pen
            X.append([raw_pen])
            y.append(float(max(0.0, min(1.0, target_w))))
        if len(X) < 3:
            return
        reg = _Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y)
        raw = float(reg.coef_[0])
        self._weight = float(max(self._CLAMP[0], min(self._CLAMP[1], raw)))
        log.info(f"[SizePenaltyWeightModel] Fitted weight={self._weight:.4f} "
                 f"({len(X)} small-dataset records)")


class ConfidencePenaltyWeightModel:
    """
    Ridge regression that learns optimal penalty weights for:
      - variance penalty  (cv_std-related)
      - size penalty      (small dataset)
      - overfitting penalty

    Trained from the ledger: target = observed gap between CV score and
    actual test performance.  The three binary/continuous penalty flags
    are regressed against this gap to learn their individual contributions.

    Replaces all three hardcoded scalar multipliers in calculate_confidence().
    """

    MIN_SAMPLES = 25
    RETRAIN_EVERY = 10

    _DEFAULTS = {
        "variance_scale": 2.0,      # (cv_std - 0.05) * variance_scale
        "overfit_weight": 0.20,     # penalty when overfitting detected
        "size_base":      0.25,     # (1 - n/200) * size_base
    }
    _CLAMPS = {
        "variance_scale": (0.5,  5.0),
        "overfit_weight": (0.05, 0.40),
        "size_base":      (0.05, 0.50),
    }

    def __init__(self) -> None:
        self._weights: dict = dict(self._DEFAULTS)
        self._next_retrain: int = self.MIN_SAMPLES

    def get(self, key: str) -> float:
        return self._weights.get(key, self._DEFAULTS[key])

    def maybe_fit(self, ledger_records: list) -> None:
        if len(ledger_records) >= self._next_retrain:
            self._fit(ledger_records)
            self._next_retrain = len(ledger_records) + self.RETRAIN_EVERY

    def _fit(self, records: list) -> None:
        try:
            from sklearn.linear_model import Ridge as _Ridge
        except ImportError:
            return
        X, y = [], []
        for r in records:
            perf = r.get("metrics") or r.get("performance") or {}
            cv   = float(perf.get("cv_score_mean", 0.5) or 0.5)
            test = float(perf.get("accuracy") or perf.get("R2") or cv)
            cv_std = float(perf.get("cv_score_std", 0.0) or 0.0)
            overfit = float(bool(perf.get("overfitting", False)))
            n_rows  = float((r.get("dataset_diagnostics") or {}).get("n_rows", 9999))
            gap     = max(0.0, cv - test)
            # feature components
            var_comp  = max(0.0, cv_std - 0.05)
            size_comp = max(0.0, 1.0 - n_rows / 200.0)
            X.append([var_comp, size_comp, overfit])
            y.append(gap)
        if len(X) < 5:
            return
        reg = _Ridge(alpha=1.0, fit_intercept=False, positive=True)
        reg.fit(X, y)
        keys = ["variance_scale", "size_base", "overfit_weight"]
        for k, raw_coef in zip(keys, reg.coef_):
            lo, hi = self._CLAMPS[k]
            self._weights[k] = float(max(lo, min(hi, raw_coef)))
        log.info(f"[ConfidencePenaltyWeightModel] Fitted: {self._weights} "
                 f"({len(X)} records)")


# Module-level singletons — warmed up from ledger at module load
_global_size_penalty_model      = SizePenaltyWeightModel()
_global_confidence_penalty_model = ConfidencePenaltyWeightModel()


def _warm_confidence_models_from_ledger() -> None:
    """Load ledger records once at startup to warm the penalty models."""
    try:
        import json as _json
        lp = Path("agent_system_state") / "outcome_ledger.json"
        if lp.exists():
            with open(lp) as fh:
                records = _json.load(fh).get("records", [])
            _global_size_penalty_model.maybe_fit(records)
            _global_confidence_penalty_model.maybe_fit(records)
            log.info(f"[ConfidenceModels] Warm-started from {len(records)} ledger records.")
    except Exception as exc:
        log.debug(f"[ConfidenceModels] Ledger warm-start skipped: {exc}")


_warm_confidence_models_from_ledger()


# ==============================================================================
# SECTION 2.7 — META PRIORITY SCORER (v5.6)
# ==============================================================================

class MetaPriorityScorer:
    """
    GradientBoosting regressor that learns which model family produces the
    highest CV score for a given dataset profile — replacing hardcoded rules.

    The model learns from the cross-run outcome ledger (persisted by
    automl_integration) which candidates tend to win for which profiles.
    Non-obvious interactions (e.g. boosting under-performing on heavy-missing
    data even with large n_rows) emerge naturally from observed outcomes.

    Falls back to rule-based scoring when no ledger data is available.
    """

    MIN_SAMPLES   = 15
    RETRAIN_EVERY = 10

    _FAMILY_MAP = {
        "linear":   ["logisticregression","linearsvc","linearsvr","ridge",
                     "ridgeclassifier","lasso","elasticnet","bayesianridge","sgd"],
        "tree":     ["randomforest","extratrees","decisiontree","gradientboosting",
                     "histgradientboosting","adaboost","bagging"],
        "boosting": ["xgboost","lightgbm","catboost"],
        "nb":       ["gaussiannb","bernoullinb","multinomialnb","qda"],
        "knn":      ["kneighbors"],
    }

    def __init__(self) -> None:
        self._reg:          object = None
        self._next_retrain: int    = self.MIN_SAMPLES

    def maybe_fit(self, ledger_records: list) -> None:
        if len(ledger_records) >= self._next_retrain:
            self._fit(ledger_records)
            self._next_retrain = len(ledger_records) + self.RETRAIN_EVERY

    def score(self, model_name: str, n_rows: float,
              n_features: float, problem_type: str,
              missing_pct: float = 0.0) -> Optional[float]:
        if self._reg is None:
            return None
        try:
            vec = self._build_vec(model_name, n_rows, n_features, problem_type, missing_pct)
            return float(self._reg.predict([vec])[0])
        except Exception:
            return None

    def _name_to_family_idx(self, name: str) -> float:
        lc = name.lower().replace("_", "").replace(" ", "")
        family_idx = {"linear": 0, "tree": 1, "boosting": 2, "nb": 3, "knn": 4}
        for fam, kws in self._FAMILY_MAP.items():
            if any(kw in lc for kw in kws):
                return float(family_idx.get(fam, 5)) / 5.0
        return 1.0

    def _build_vec(self, name: str, n_rows: float, n_features: float,
                   problem_type: str, missing_pct: float = 0.0) -> list:
        return [
            math.log10(max(n_rows,     1) + 1) / 7.0,
            math.log10(max(n_features, 1) + 1) / 4.0,
            min(n_features / max(n_rows, 1), 2.0),
            1.0 if problem_type == "regression" else 0.0,
            self._name_to_family_idx(name),
            min(float(missing_pct) / 100.0, 1.0),
        ]

    def _fit(self, records: list) -> None:
        try:
            from sklearn.ensemble import GradientBoostingRegressor as _GBR
        except ImportError:
            return
        X, y = [], []
        for r in records:
            model  = r.get("best_model") or r.get("best_model_name", "")
            metrics = r.get("metrics") or r.get("performance") or {}
            cv     = metrics.get("cv_score_mean")
            diag   = r.get("dataset_diagnostics") or {}
            n_rows = float(diag.get("n_rows", 0) or 0)
            n_feat = float(diag.get("n_cols", 0) or 0)
            ptype  = r.get("problem_type", "classification")
            miss   = float(diag.get("overall_missing_pct", 0) or 0)
            if not model or cv is None:
                continue
            try:
                cv_f = float(cv)
                if math.isnan(cv_f) or math.isinf(cv_f):
                    continue
                X.append(self._build_vec(model, n_rows, n_feat, ptype, miss))
                y.append(cv_f)
            except (TypeError, ValueError):
                continue
        if len(X) < 5:
            return
        reg = _GBR(n_estimators=60, max_depth=3, learning_rate=0.1,
                   random_state=42, min_samples_leaf=2)
        reg.fit(X, y)
        self._reg = reg
        log.info(f"[MetaPriorityScorer] Fitted on {len(X)} ledger records.")


_global_priority_scorer = MetaPriorityScorer()


def _warm_priority_scorer_from_ledger() -> None:
    try:
        import json as _json
        lp = Path("agent_system_state") / "outcome_ledger.json"
        if lp.exists():
            with open(lp) as fh:
                records = _json.load(fh).get("records", [])
            _global_priority_scorer.maybe_fit(records)
    except Exception as exc:
        log.debug(f"[MetaPriorityScorer] Ledger warm-start skipped: {exc}")


_warm_priority_scorer_from_ledger()


# ==============================================================================
# SECTION 6.7 — RL AGENT UTILITIES
# ==============================================================================

def detect_data_leakage(X, y, threshold: float = 0.90) -> tuple:
    removed_features: list = []
    try:
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().all():
            return X, removed_features, False
        for col in X.columns:
            try:
                corr = X[col].corr(y_num)
                if pd.notna(corr) and abs(corr) > threshold:
                    removed_features.append(col)
                    log.warning(f"[LeakageGuard] Removing '{col}' — |r|={abs(corr):.3f}")
            except Exception:
                pass
        X_clean = X.drop(columns=removed_features, errors="ignore") if removed_features else X
        return X_clean, removed_features, bool(removed_features)
    except Exception as exc:
        log.warning(f"[LeakageGuard] Check failed: {exc}")
        return X, [], False


def detect_overfitting(train_score: float, cv_score: float,
                       gap_threshold: float = 0.10) -> bool:
    if train_score is None or cv_score is None:
        return False
    gap = train_score - cv_score
    is_overfit = gap > gap_threshold
    if is_overfit:
        log.warning(f"[Overfitting] Detected — gap={gap:.4f}")
    return is_overfit


def calculate_confidence(
    cv_score:         float,
    cv_std:           float,
    dataset_size:     int,
    overfitting_flag: bool,
) -> dict:
    """
    v5.6 — Fully hybrid confidence scoring.

    All three penalty components now use LEARNED weights from
    ConfidencePenaltyWeightModel, which regresses observed CV→test
    performance gaps against each penalty's raw feature component.

    Architecture
    ────────────
    base = clamp(cv_score, 0, 1)

    variance_pen = max(0, cv_std - 0.05) * learned_variance_scale
    size_pen     = max(0, 1 - n/200)     * learned_size_base
    overfit_pen  = learned_overfit_weight  (if flag set)

    hybrid_total = base - variance_pen - size_pen - overfit_pen

    The SizePenaltyWeightModel provides a secondary size weight estimate
    as a cross-check; the average is used.

    All weights clamp to [0.05, 0.50] so the system can never zero out
    a penalty entirely (safety) or apply an extreme penalty (stability).
    """
    # Re-warm models in case new ledger data arrived since module load
    try:
        import json as _json
        lp = Path("agent_system_state") / "outcome_ledger.json"
        if lp.exists():
            with open(lp) as fh:
                _records = _json.load(fh).get("records", [])
            _global_confidence_penalty_model.maybe_fit(_records)
            _global_size_penalty_model.maybe_fit(_records)
    except Exception:
        pass

    base    = max(0.0, min(1.0, float(cv_score or 0)))
    penalty = 0.0

    # ── Variance penalty — learned scale factor ───────────────────────────────
    if cv_std is not None and cv_std > 0.05:
        var_scale = _global_confidence_penalty_model.get("variance_scale")
        var_pen   = min(0.30, (cv_std - 0.05) * var_scale)
        penalty  += var_pen
        log.info(f"[Confidence] −{var_pen:.3f} variance (std={cv_std:.4f}, "
                 f"learned_scale={var_scale:.2f})")

    # ── Size penalty — HYBRID: physical formula × learned weight ─────────────
    # Physical: larger raw_pen = more uncertain (unavoidable)
    # Learned:  how much does size actually hurt in this system's history?
    if dataset_size is not None and dataset_size < 200:
        raw_pen   = max(0.0, 1.0 - dataset_size / 200.0)

        # Cross-check both learned estimates and average for stability
        w1        = _global_confidence_penalty_model.get("size_base")
        w2        = _global_size_penalty_model.weight
        size_w    = (w1 + w2) / 2.0
        size_pen  = min(0.30, raw_pen * size_w)
        penalty  += size_pen
        log.info(f"[Confidence] −{size_pen:.3f} size ({dataset_size} rows, "
                 f"learned_w={size_w:.3f})")

    # ── Overfitting penalty — learned weight ──────────────────────────────────
    if overfitting_flag:
        # Primary: ConfidencePenaltyWeightModel
        overfit_w = _global_confidence_penalty_model.get("overfit_weight")
        # Cross-check: rl_agent PenaltyWeightModel if available
        try:
            from rl_agent import _global_penalty_model as _rl_pm
            rl_w      = _rl_pm._weights.get("overfitting", overfit_w)
            overfit_w = (overfit_w + rl_w) / 2.0   # blend both estimates
        except ImportError:
            pass
        penalty  += overfit_w
        log.info(f"[Confidence] −{overfit_w:.3f} overfitting (learned)")

    score = round(max(0.0, base - penalty), 4)
    label = "High" if score >= 0.75 else ("Medium" if score >= 0.50 else "Low")
    log.info(f"[Confidence] final={score} ({label})  base={base:.4f}  pen={penalty:.4f}")
    return {"score": score, "label": label}


# ==============================================================================
# SECTION 6.75 — MULTI-ARMED BANDIT ENGINE
# ==============================================================================

def _build_rl_state(X_train, y_train, problem_type, tier, n_candidates) -> dict:
    rows     = int(len(X_train))
    features = int(X_train.shape[1])
    try:
        missing_ratio = round(float(X_train.isnull().values.mean()), 4)
    except Exception:
        missing_ratio = 0.0
    class_imbalance: Optional[float] = None
    target_std:      Optional[float] = None
    if problem_type == "classification":
        try:
            class_imbalance = round(float(y_train.value_counts(normalize=True).max()), 4)
        except Exception:
            pass
    else:
        try:
            target_std = round(float(y_train.std()), 4)
        except Exception:
            pass
    return {
        "rows":            rows,
        "features":        features,
        "missing_ratio":   missing_ratio,
        "class_imbalance": class_imbalance,
        "target_std":      target_std,
        "n_candidates":    n_candidates,
        "tier":            tier,
        "problem_type":    problem_type,
    }


def _init_model_stats(candidates: dict) -> dict:
    return {name: {"trials": 0, "avg_score": 0.0} for name in candidates}


def select_model_epsilon_greedy(
    model_stats, epsilon=0.20, iteration=0, n_iterations=10,
    consecutive_counts=None, last_chosen=None, max_consecutive=3,
) -> str:
    names       = list(model_stats.keys())
    all_untried = all(s["trials"] == 0 for s in model_stats.values())
    if all_untried:
        top_k  = max(1, len(names) // 2 + 1)
        chosen = random.choice(names[:top_k])
        return chosen
    decay_t       = iteration / max(n_iterations - 1, 1)
    eps_high      = min(0.80, epsilon * 2)
    eps_low       = max(0.05, epsilon / 2)
    epsilon_decay = eps_high - (eps_high - eps_low) * decay_t
    untried = [n for n in names if model_stats[n]["trials"] == 0]
    if untried:
        return untried[0]
    if (consecutive_counts is not None and last_chosen is not None and
            consecutive_counts.get(last_chosen, 0) >= max_consecutive):
        pool = [n for n in names if n != last_chosen]
        if pool:
            return random.choice(pool)
    if random.random() < epsilon_decay:
        return random.choice(names)
    total_trials = sum(s["trials"] for s in model_stats.values())
    tried    = {n: s for n, s in model_stats.items() if s["trials"] > 0}
    best_avg = max(s["avg_score"] for s in tried.values())
    near_best = [n for n, s in tried.items() if best_avg - s["avg_score"] <= 0.005]
    if len(near_best) > 1:
        return max(near_best, key=lambda n: _ucb1_score(
            tried[n]["avg_score"], tried[n]["trials"], total_trials))
    return max(tried, key=lambda n: tried[n]["avg_score"])


def update_model_stats(model_stats: dict, model_name: str, score: float) -> None:
    s = model_stats[model_name]
    s["trials"]    += 1
    s["avg_score"] += (score - s["avg_score"]) / s["trials"]


_RL_EPS_START    = 0.40
_RL_EPS_END      = 0.10
_RL_UCB_C        = 1.41
_RL_MAX_CONSEC   = 3
_OVERFIT_PENALTY = 0.15
_STD_PENALTY_K   = 0.50


def _ucb1_score(avg_score, trials, total_trials, c=1.41) -> float:
    if trials == 0:
        return float("inf")
    return avg_score + c * math.sqrt(math.log(max(total_trials, 1)) / trials)


def _compute_effective_score(score, std, in_loop_overfit,
                              overfit_gap=0.0, gap_threshold=0.10) -> float:
    penalty = 0.0
    if in_loop_overfit or overfit_gap > 0:
        factor  = min(1.0, overfit_gap / max(gap_threshold, 1e-9))
        penalty += _OVERFIT_PENALTY * factor
    if std is not None and std > 0.05:
        penalty += _STD_PENALTY_K * (std - 0.05)
    return max(0.0, score - penalty)


def _check_baseline(best_name, cv_scores, baseline_score, problem_type,
                    margin=0.01) -> dict:
    if baseline_score is None:
        baseline_score = 0.0 if problem_type == "regression" else None
    if baseline_score is None:
        return {"triggered": False}
    best_cv = cv_scores.get(best_name, 0.0) or 0.0
    gap = best_cv - baseline_score
    alert = {
        "triggered": gap <= margin,
        "best_cv":   round(best_cv, 4),
        "baseline":  round(baseline_score, 4),
        "gap":       round(gap, 4),
        "margin":    margin,
        "message":   "",
    }
    if alert["triggered"]:
        alert["message"] = (
            f"BASELINE_ALERT: Best model '{best_name}' CV={best_cv:.4f} is only "
            f"{gap*100:.2f}pp above baseline ({baseline_score:.4f}).")
        log.warning(f"[BaselineAlert] {alert['message']}")
    return alert


def _get_train_score(name, base_est, best_params, X_train, y_train, scoring):
    try:
        model = _build_final_model(name, base_est, best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        if scoring in ("f1", "f1_macro"):
            return float(f1_score(y_train, preds, average="macro", zero_division=0))
        elif scoring == "accuracy":
            return float(accuracy_score(y_train, preds))
        else:
            return float(_safe_float(r2_score(y_train, preds)) or 0.0)
    except Exception:
        return None


def run_rl_pipeline(X_train, X_test, y_train, y_test, candidates, problem_type,
                    tier, cv_splitter, scoring, n_iterations=None, epsilon=0.20,
                    rl_state=None) -> tuple:
    n_candidates = len(candidates)
    if n_iterations is None:
        _imap = {1: min(15, n_candidates*2), 2: min(10, n_candidates*2),
                 3: min(8, n_candidates),    4: min(5, n_candidates)}
        n_iterations = _imap.get(tier, min(10, n_candidates))

    model_stats        = _init_model_stats(candidates)
    agent_history      = []
    cv_scores          = {}
    cv_std_scores      = {}
    best_params_all    = {}
    consecutive_counts = {n: 0 for n in candidates}
    last_chosen        = None

    for iteration in range(n_iterations):
        name = select_model_epsilon_greedy(
            model_stats, epsilon=epsilon, iteration=iteration,
            n_iterations=n_iterations, consecutive_counts=consecutive_counts,
            last_chosen=last_chosen, max_consecutive=3)
        consecutive_counts[name] = (consecutive_counts.get(name, 0) + 1
                                    if name == last_chosen else 1)
        last_chosen = name
        base_est, param_grid = candidates[name]
        try:
            score, std, bp = _search_hyperparams(
                name, base_est, param_grid, X_train, y_train,
                cv_splitter, scoring, tier)
            train_score     = _get_train_score(name, base_est, bp, X_train, y_train, scoring)
            overfit_gap     = max(0.0, (train_score or score) - score)
            in_loop_overfit = detect_overfitting(
                train_score if train_score is not None else score, score)
            effective_score = _compute_effective_score(
                score, std, in_loop_overfit, overfit_gap=overfit_gap)
            update_model_stats(model_stats, name, effective_score)
            if name not in cv_scores or score > cv_scores[name]:
                cv_scores[name]       = score
                cv_std_scores[name]   = std
                best_params_all[name] = bp
            _decay_t  = iteration / max(n_iterations - 1, 1)
            _eps_used = round(min(0.80, epsilon*2) -
                              (min(0.80, epsilon*2) - max(0.05, epsilon/2)) * _decay_t, 3)
            entry = {
                "iteration":       iteration + 1,
                "model":           name,
                "action":          "cold_start" if model_stats[name]["trials"] == 1 else "exploit_or_explore",
                "score":           round(float(score), 4),
                "effective_score": round(float(effective_score), 4),
                "std":             round(float(std), 4),
                "train_score":     round(float(train_score), 4) if train_score is not None else None,
                "overfit":         in_loop_overfit,
                "avg_score":       round(model_stats[name]["avg_score"], 4),
                "trials":          model_stats[name]["trials"],
                "epsilon_used":    _eps_used,
            }
            if rl_state:
                entry["state"] = rl_state
            agent_history.append(entry)
            log.info(f"[RL-Agent] [{iteration+1:02d}/{n_iterations}] "
                     f"{name:<28} cv={score:.4f}±{std:.4f} eff={effective_score:.4f}")
        except Exception as exc:
            log.warning(f"[RL-Agent] Iter {iteration+1}: '{name}' failed — {exc}")
            model_stats[name]["trials"] += 1
            entry = {"iteration": iteration+1, "model": name, "action": "failed",
                     "score": None, "effective_score": None, "train_score": None,
                     "overfit": None, "avg_score": model_stats[name]["avg_score"],
                     "trials": model_stats[name]["trials"], "error": str(exc)}
            if rl_state:
                entry["state"] = rl_state
            agent_history.append(entry)

    evaluated = {n: s for n, s in model_stats.items() if s["trials"] > 0}
    if not evaluated:
        return None, None, {}, {}, {}, agent_history, model_stats

    best_name = max(evaluated, key=lambda n: evaluated[n]["avg_score"])
    best_model = _build_final_model(
        best_name, candidates[best_name][0], best_params_all.get(best_name, {}))
    best_model.fit(X_train, y_train)
    return (best_name, best_model, cv_scores, cv_std_scores,
            best_params_all, agent_history, model_stats)


# ==============================================================================
# SECTION 6.8 — MULTI-RUN STABILITY WRAPPER
# ==============================================================================

_STABILITY_SEEDS    = [42, 0, 99]
_STABILITY_TIERS    = {1, 2}
_STABILITY_MIN_ROWS = 50


def _run_rl_multi_seed(X_train, X_test, y_train, y_test, candidates,
                       problem_type, tier, scoring, n_folds,
                       rl_state=None, seeds=None) -> tuple:
    if seeds is None:
        seeds = _STABILITY_SEEDS
    n_rows = len(X_train)
    if tier not in _STABILITY_TIERS or n_rows < _STABILITY_MIN_ROWS or len(seeds) < 2:
        return run_rl_pipeline(
            X_train, X_test, y_train, y_test, candidates, problem_type, tier,
            (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seeds[0])
             if problem_type == "classification"
             else KFold(n_splits=n_folds, shuffle=True, random_state=seeds[0])),
            scoring, rl_state=rl_state)

    all_cv_scores, all_cv_std, all_params, all_history = [], [], [], []
    last_stats, last_best_model, last_best_name = {}, None, ""
    for seed_idx, seed in enumerate(seeds):
        random.seed(seed)
        cv_split = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    if problem_type == "classification"
                    else KFold(n_splits=n_folds, shuffle=True, random_state=seed))
        try:
            (s_best_name, s_best_model, s_cv_scores, s_cv_std,
             s_params, s_history, s_stats) = run_rl_pipeline(
                X_train, X_test, y_train, y_test, candidates,
                problem_type, tier, cv_split, scoring, rl_state=rl_state)
        except Exception as exc:
            log.warning(f"[Stability] Seed {seed} failed: {exc}")
            continue
        for entry in s_history:
            entry["seed"] = seed; entry["seed_idx"] = seed_idx
        all_history.extend(s_history)
        all_cv_scores.append(s_cv_scores); all_cv_std.append(s_cv_std)
        all_params.append(s_params); last_stats = s_stats
        last_best_model = s_best_model; last_best_name = s_best_name

    if not all_cv_scores:
        return None, None, {}, {}, {}, all_history, {}

    all_models = set()
    for d in all_cv_scores: all_models.update(d.keys())
    mean_scores, pooled_std = {}, {}
    for model in all_models:
        seed_scores = [d[model] for d in all_cv_scores if model in d]
        seed_stds   = [d.get(model, 0.0) for d in all_cv_std if model in d]
        if not seed_scores: continue
        mean_scores[model] = float(np.mean(seed_scores))
        pooled_std[model]  = float(np.sqrt(
            float(np.mean([s**2 for s in seed_stds])) + float(np.var(seed_scores))))

    if not mean_scores:
        return None, None, {}, {}, {}, all_history, {}

    stable_winner = max(mean_scores, key=lambda n: mean_scores[n])
    best_params_merged = {}
    for d in all_params:
        if stable_winner in d:
            best_params_merged = dict(d[stable_winner]); break

    if stable_winner != last_best_name and stable_winner in candidates:
        final_model = _build_final_model(
            stable_winner, candidates[stable_winner][0], best_params_merged)
        final_model.fit(X_train, y_train)
    else:
        final_model = last_best_model

    all_history.append({"stability_report": {
        "n_seeds": len(seeds), "seeds": seeds,
        "mean_scores": {m: round(s, 4) for m, s in mean_scores.items()},
        "pooled_std":  {m: round(s, 4) for m, s in pooled_std.items()},
        "stable_winner": stable_winner}})
    return (stable_winner, final_model, mean_scores, pooled_std,
            best_params_merged if isinstance(best_params_merged, dict) else {},
            all_history, last_stats)


# ==============================================================================
# SECTION 3 — SPECIAL CHARACTER CLEANING  (unchanged from v5.5)
# ==============================================================================

_PHONE_PAT = re.compile(r"^[\+]?[\d][\d\s\-\.\(\)\/\\]{5,18}$")
_PRICE_PAT = re.compile(
    r"^[\$\£\€\¥\₹\₩\₽\₺\฿\₴\₦\₱\₲\₡\₵\₸\₫\₭\₮\﷼\؋\৳\රු\元\円\د\.إ\ر\.س]?"
    r"\s*[\d][\d\s,\.']*[\d]?\s*"
    r"[\$\£\€\¥\₹\₩\₽\₺\฿\₴\₦\₱\₲\₡\₵\₸\₫\₭\₮\﷼\؋\৳\රු\元\円]?"
    r"\s*%?$", re.UNICODE)
_POSTCODE_PAT = re.compile(
    r"^(?:\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}"
    r"|[A-Z]\d[A-Z]\s?\d[A-Z]\d|〒?\d{3}-?\d{4}|\d{3}-?\d{3}"
    r"|\d{4,6}|\d{5}-\d{3})$", re.IGNORECASE)
_ID_DASH_PAT    = re.compile(r"^[A-Za-z0-9]+(?:[-_\.][A-Za-z0-9]+)+$")
_EMAIL_PAT      = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_URL_PAT        = re.compile(r"^(?:https?://|ftp://|www\.)[^\s]{4,}$", re.I)
_IP_PAT         = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?$")
_COMPOUND_SEPS  = ["-", "/", "|", ";", "~", "::"]
_FREE_EMAIL     = {"gmail","yahoo","hotmail","outlook","icloud","live","aol",
                   "protonmail","zoho","yandex","mail","inbox","gmx","tutanota"}
_NATIONAL_ID_PATTERNS = [
    ("USA_SSN",     re.compile(r"^\d{3}-\d{2}-\d{4}$")),
    ("IND_AADHAAR", re.compile(r"^\d{4}[\s\-]\d{4}[\s\-]\d{4}$")),
    ("IND_PAN",     re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")),
    ("CHN_ID",      re.compile(r"^\d{17}[\dXx]$")),
    ("GBR_NIN",     re.compile(r"^[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]$", re.I)),
]
_ID_COL_KEYWORDS = ("ssn","sin","nin","national_id","id_no","passport","license",
                    "aadhaar","aadhar","pan","cpf","tfn","rfc","eid","taxid")
_DATE_COL_KEYWORDS = ("date","time","datetime","timestamp","dob","birth",
                      "hired","joined","created","updated","start","end",
                      "expir","due","registered","signup","login","modified")
_BINARY_MAPS = [
    ({"yes","no"},           {"yes":1,"no":0}),
    ({"true","false"},       {"true":1,"false":0}),
    ({"y","n"},              {"y":1,"n":0}),
    ({"1","0"},              {"1":1,"0":0}),
    ({"on","off"},           {"on":1,"off":0}),
    ({"active","inactive"},  {"active":1,"inactive":0}),
    ({"pass","fail"},        {"pass":1,"fail":0}),
    ({"male","female"},      {"male":1,"female":0}),
    ({"present","absent"},   {"present":1,"absent":0}),
    ({"open","closed"},      {"open":1,"closed":0}),
]
_ORDINAL_MAPS = [
    ["very low","low","medium","high","very high"],
    ["very poor","poor","average","good","excellent"],
    ["strongly disagree","disagree","neutral","agree","strongly agree"],
    ["never","rarely","sometimes","often","always"],
    ["none","low","medium","high","critical"],
    ["beginner","intermediate","advanced","expert"],
    ["bronze","silver","gold","platinum","diamond"],
    ["cold","warm","hot"],
    ["slow","moderate","fast"],
    ["small","medium","large","xlarge"],
    ["s","m","l","xl","xxl"],
    ["q1","q2","q3","q4"],
    ["very dissatisfied","dissatisfied","neutral","satisfied","very satisfied"],
]
_POS_WORDS = {"good","great","excellent","amazing","wonderful","fantastic","best",
              "positive","happy","success","win","love","perfect","outstanding"}
_NEG_WORDS = {"bad","poor","terrible","awful","worst","negative","fail","failed",
              "error","broken","missing","lost","dead","inactive","disabled"}
_MISSING_TOKENS = {"nan","none","null","na","n/a","","?","-","--"}
_UNIT_PATTERNS = [
    (re.compile(r"^\s*\$?\s*([\d,\.]+)\s*[Bb][Nn]?\b"),   1_000_000_000),
    (re.compile(r"^\s*\$?\s*([\d,\.]+)\s*[Mm][Nn]?\b"),   1_000_000),
    (re.compile(r"^\s*\$?\s*([\d,\.]+)\s*[Kk]\b"),        1_000),
    (re.compile(r"^\s*([\d\.]+)\s*(?:years?|yrs?)\s*$", re.I), 1),
    (re.compile(r"^\s*([\d\.]+)\s*(?:kg|KG|Kg)\s*$"),     1),
    (re.compile(r"^\s*([\d\.]+)\s*(?:lbs?|LBS?)\s*$"),    1),
]
_DOMAIN_RANGES = [
    ("latitude",   -90.0,    90.0),
    ("longitude", -180.0,   180.0),
    ("age",          0.0,   130.0),
    ("bmi",         10.0,    80.0),
    ("humidity",     0.0,   100.0),
    ("credit_score",300.0,   850.0),
    ("rating",       0.0,    10.0),
    ("percent",      0.0,   100.0),
    ("probability",  0.0,     1.0),
]


def _col_match_rate(series, pattern, threshold=0.60):
    sample = series.dropna().astype(str).str.strip().head(200)
    return len(sample) > 0 and sample.str.match(pattern).mean() >= threshold

def _detect_national_id(series, threshold=0.50):
    sample = series.dropna().astype(str).str.strip().head(200)
    if len(sample) == 0: return None
    for name, pat in _NATIONAL_ID_PATTERNS:
        if sample.str.match(pat).mean() >= threshold: return name
    return None

def _normalise_price_string(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d\s,\.\'\-]", "", regex=True).str.strip()
    def _fix(val):
        val = val.strip()
        if not val: return val
        if re.search(r",\d{2}$", val) and "." in val:
            val = val.replace(".", "").replace(",", ".")
        elif re.search(r",\d{2}$", val):
            val = val.replace(",", ".")
        else:
            val = val.replace(",", "").replace("'", "").replace(" ", "")
        return val
    return s.apply(_fix).replace("", np.nan)

def _normalise_text_column(series):
    cleaned = series.astype(str).str.strip()
    if cleaned.nunique() <= 50: cleaned = cleaned.str.lower()
    return cleaned.replace({"nan": np.nan, "none": np.nan, "None": np.nan, "": np.nan})

def _strip_units(series):
    def _try(val):
        if not isinstance(val, str):
            try: return float(val)
            except: return None
        val = val.strip()
        for pat, mult in _UNIT_PATTERNS:
            m = pat.fullmatch(val)
            if m:
                try: return float(m.group(1).replace(",","")) * mult
                except: return None
        return None
    sample = series.dropna().astype(str).str.strip().head(300)
    if len(sample) == 0: return series
    if sample.apply(_try).notna().mean() >= 0.60:
        return pd.to_numeric(series.astype(str).str.strip().apply(_try), errors="coerce")
    return series

def _parse_ratio(val):
    if not isinstance(val, str): return None
    m = re.match(r"^([\d\.]+)\s*/\s*[\d\.]+$", val.strip())
    if m:
        try: return float(m.group(1))
        except: pass
    return None

def _encode_binary_text(df, target_column):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            lowered = df[col].dropna().astype(str).str.strip().str.lower()
            unique_vals = set(lowered.unique())
            if len(unique_vals) == 0 or len(unique_vals) > 3: continue
            for known_set, mapping in _BINARY_MAPS:
                if unique_vals <= (known_set | {"nan","none","null",""}):
                    df[col] = (df[col].astype(str).str.strip().str.lower()
                               .map(mapping).where(df[col].notna(), other=np.nan))
                    break
        except Exception: pass
    return df

def _encode_ordinal_columns(df, target_column):
    ordinal_encoded = set()
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            lowered    = df[col].dropna().astype(str).str.strip().str.lower()
            unique_vals = set(lowered.unique())
            if len(unique_vals) == 0: continue
            for ordinal_scale in _ORDINAL_MAPS:
                scale_set = set(ordinal_scale)
                if unique_vals <= scale_set and len(unique_vals) >= 2:
                    rank_map = {v: i for i, v in enumerate(ordinal_scale)}
                    df[col] = (df[col].astype(str).str.strip().str.lower()
                               .map(rank_map).where(df[col].notna(), other=np.nan))
                    ordinal_encoded.add(col)
                    break
        except Exception: pass
    return df, ordinal_encoded

def _split_compound_columns(df, target_column):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        if any(k in col.lower() for k in _DATE_COL_KEYWORDS): continue
        try:
            series = df[col].dropna().astype(str).str.strip()
            if len(series) == 0: continue
            for sep in _COMPOUND_SEPS:
                hit_rate = series.str.contains(re.escape(sep), regex=False).mean()
                if hit_rate < 0.70: continue
                split_df = series.str.split(sep, n=1, expand=True)
                if split_df.shape[1] < 2: continue
                p1 = split_df[0].str.strip(); p2 = split_df[1].str.strip()
                n  = len(series)
                if p1.nunique()/n > 0.95 or p2.nunique()/n > 0.95: continue
                full_split = df[col].astype(str).str.strip().str.split(sep, n=1, expand=True)
                new_p1 = col + "_part1"; new_p2 = col + "_part2"
                df[new_p1] = full_split[0].str.strip().replace({"nan": np.nan, "": np.nan})
                df[new_p2] = (full_split[1].str.strip().replace({"nan": np.nan, "": np.nan})
                              if full_split.shape[1] > 1 else np.nan)
                df.drop(columns=[col], inplace=True)
                break
        except Exception: pass
    return df

def _extract_email_features(df, target_column):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            sample = df[col].dropna().astype(str).str.strip().head(200)
            if len(sample) == 0 or sample.str.match(_EMAIL_PAT).mean() < 0.70: continue
            def _domain(val):
                try: dom = val.strip().lower().split("@")[1]; return dom.split(".")[0]
                except: return "unknown"
            domains = df[col].astype(str).apply(_domain)
            df[col + "_domain"]  = domains.where(df[col].notna(), other=np.nan)
            df[col + "_is_free"] = domains.isin(_FREE_EMAIL).astype(int)
            df.drop(columns=[col], inplace=True)
        except Exception: pass
    return df

def _handle_url_ip_columns(df, target_column):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            sample = df[col].dropna().astype(str).str.strip().head(200)
            if len(sample) == 0: continue
            if sample.str.match(_IP_PAT).mean() >= 0.70:
                df.drop(columns=[col], inplace=True, errors="ignore"); continue
            if sample.str.match(_URL_PAT).mean() >= 0.70:
                def _url_domain(val):
                    try:
                        v = re.sub(r"^https?://|^ftp://|^www\.", "", val.strip().lower())
                        dom = v.split("/")[0].split(".")
                        return dom[-2] if len(dom) >= 2 else dom[0]
                    except: return "unknown"
                df[col + "_domain"] = df[col].astype(str).apply(_url_domain)
                df.drop(columns=[col], inplace=True, errors="ignore")
        except Exception: pass
    return df

def _extract_text_features(df, target_column):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            sample = df[col].dropna().astype(str).str.strip().head(200)
            if len(sample) == 0: continue
            if sample.str.split().str.len().mean() < 4 or sample.str.len().mean() < 20: continue
            full = df[col].fillna("").astype(str).str.strip().str.lower()
            words_series = full.str.split()
            df[col + "_word_count"]  = words_series.str.len().fillna(0).astype(int)
            df[col + "_char_length"] = full.str.len().astype(int)
            def _sentiment(words):
                if not isinstance(words, list) or len(words) == 0: return 0.0
                pos = sum(1 for w in words if w in _POS_WORDS)
                neg = sum(1 for w in words if w in _NEG_WORDS)
                return round((pos - neg) / len(words), 4)
            df[col + "_sentiment"] = words_series.apply(_sentiment)
            df.drop(columns=[col], inplace=True, errors="ignore")
        except Exception: pass
    return df

def _validate_phone_columns(df, target_column):
    phone_kw = ("phone","mobile","tel","cell","fax","contact")
    for col in list(df.columns):
        if col == target_column: continue
        if not any(k in col.lower() for k in phone_kw): continue
        try:
            numeric_vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(numeric_vals) == 0: continue
            if (numeric_vals < 0).mean() >= 0.30 or \
               (numeric_vals.astype(str).str.replace("-","").str.len() < 6).mean() >= 0.30:
                df.drop(columns=[col], inplace=True, errors="ignore")
        except Exception: pass
    return df

def _parse_capped_numerics(df, target_column):
    _cap_pat   = re.compile(r"^[<>≤≥~+\-±]?\s*([\d,\.]+)\s*[+]?$")
    _range_pat = re.compile(r"^([\d,\.]+)\s*[-–—to]+\s*([\d,\.]+)$", re.I)
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        try:
            sample = df[col].dropna().astype(str).str.strip().head(300)
            if len(sample) == 0: continue
            def _try_cap(val):
                v = val.strip()
                m = _range_pat.fullmatch(v)
                if m:
                    try: return (float(m.group(1).replace(",","")) +
                                 float(m.group(2).replace(",",""))) / 2.0
                    except: pass
                m = _cap_pat.fullmatch(v)
                if m:
                    try: return float(m.group(1).replace(",",""))
                    except: pass
                try: return float(v.replace(",",""))
                except: return None
            converted = sample.apply(_try_cap)
            nn = df[col].notna().sum()
            if nn > 0 and converted.notna().mean() >= 0.70:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().apply(_try_cap), errors="coerce")
        except Exception: pass
    return df


def _word_to_num(text):
    _ONES = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
             "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
             "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
             "nineteen":19,"twenty":20}
    _TENS = {"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,
             "eighty":80,"ninety":90}
    _MAGNITUDES = {"hundred":100,"thousand":1_000,"million":1_000_000}
    if text is None: return None
    if isinstance(text, bool): return float(text)
    if isinstance(text, (int, float)):
        v = float(text); return None if (math.isnan(v) or math.isinf(v)) else v
    try:
        v = float(text); return None if (math.isnan(v) or math.isinf(v)) else v
    except (ValueError, TypeError): pass
    if not isinstance(text, str): return None
    t = text.strip().lower()
    if t in _MISSING_TOKENS: return None
    t = re.sub(r"[^a-z0-9\s\-]", " ", t).strip()
    try: return float(t)
    except ValueError: pass
    tokens = [tok for tok in t.split() if tok]
    if not tokens: return None
    current, result, found = 0, 0, False
    for tok in tokens:
        if re.fullmatch(r"\d+", tok): current += int(tok); found = True
        elif tok in _ONES: current += _ONES[tok]; found = True
        elif tok in _TENS: current += _TENS[tok]; found = True
        elif tok in _MAGNITUDES:
            mag = _MAGNITUDES[tok]
            if mag >= 1_000: result += (current if current else 1) * mag; current = 0
            else: current = (current if current else 1) * mag
            found = True
        else: return None
    return float(result + current) if found else None


def convert_word_numbers(df, target_column, threshold=0.70):
    for col in df.select_dtypes(include=["object","category"]).columns.tolist():
        if col == target_column: continue
        if any(k in col.lower() for k in _DATE_COL_KEYWORDS): continue
        try:
            non_null = df[col].dropna()
            if len(non_null) == 0: continue
            sample = non_null.astype(str).str.strip().head(500)
            conv_sample = sample.apply(_word_to_num)
            if conv_sample.notna().mean() >= threshold:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().apply(_word_to_num), errors="coerce")
        except Exception: pass
    return df


# ==============================================================================
# SECTION 5 — UNIVERSAL CLEANING PIPELINE
# ==============================================================================

def universal_cleaning(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    ordinal_encoded_cols: set = set()
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = (df[col].astype(str).str.strip()
                       .replace({"nan": np.nan, "none": np.nan, "None": np.nan, "": np.nan}))
        except Exception: pass
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.info(f"[Clean] Removed {before - len(df)} duplicate rows.")
    df = df.dropna(axis=1, how="all")
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df.drop(columns=const_cols, inplace=True, errors="ignore")
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col == target_column: continue
        try: df[col] = _normalise_text_column(df[col])
        except Exception: pass
    _num_sentinels = {9999,-9999,999,-999,99999,-99999}
    _str_sentinels = {"error","err","#n/a","#value!","missing","n/a","na","nil","null","nan","inf","-inf"}
    for col in df.columns:
        if col == target_column: continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace(list(_num_sentinels), np.nan)
            else:
                mask = df[col].astype(str).str.strip().str.lower().isin(_str_sentinels)
                df.loc[mask, col] = np.nan
        except Exception: pass
    df = _encode_binary_text(df, target_column)
    df, ordinal_encoded_cols = _encode_ordinal_columns(df, target_column)
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col == target_column: continue
        try:
            converted = _strip_units(df[col])
            if pd.api.types.is_numeric_dtype(converted): df[col] = converted
        except Exception: pass
    df = _parse_capped_numerics(df, target_column)
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col == target_column: continue
        try:
            sample = df[col].dropna().astype(str).str.strip().head(200)
            converted = sample.apply(_parse_ratio)
            if len(sample) > 0 and converted.notna().mean() >= 0.60:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().apply(_parse_ratio), errors="coerce")
        except Exception: pass
    df = _extract_email_features(df, target_column)
    df = _handle_url_ip_columns(df, target_column)
    df = _extract_text_features(df, target_column)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col == target_column: continue
        if pd.api.types.is_datetime64_any_dtype(df[col]): continue
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            non_null  = df[col].notna().sum()
            if non_null > 0 and converted.notna().sum() / non_null >= 0.80:
                df[col + "_year"]    = converted.dt.year
                df[col + "_month"]   = converted.dt.month
                df[col + "_day"]     = converted.dt.day
                df[col + "_weekday"] = converted.dt.weekday
                if converted.dt.hour.nunique() > 1:
                    df[col + "_hour"] = converted.dt.hour
                df.drop(columns=[col], inplace=True)
        except Exception: pass
    df = _split_compound_columns(df, target_column)
    df = convert_word_numbers(df, target_column=target_column, threshold=0.70)
    if len(df) >= 50:
        hc_cols = [c for c in df.columns
                   if c != target_column and df[c].dtype == object
                   and df[c].nunique() / len(df) > 0.95]
        if hc_cols:
            df.drop(columns=hc_cols, inplace=True, errors="ignore")
    df = _validate_phone_columns(df, target_column)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col == target_column: continue
        try:
            coerced  = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if non_null > 0 and coerced.notna().sum() / non_null >= 0.90:
                df[col] = coerced
        except Exception: pass
    clamped_cols: set = set()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column or col in ordinal_encoded_cols: continue
        col_lower = col.lower()
        for keyword, lo, hi in _DOMAIN_RANGES:
            if keyword in col_lower:
                df[col] = df[col].clip(lower=lo, upper=hi)
                clamped_cols.add(col); break
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column or col in clamped_cols: continue
        try:
            s = df[col].dropna()
            if len(s) < 20: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
            if iqr == 0: continue
            df[col] = df[col].clip(lower=q1-3*iqr, upper=q3+3*iqr)
        except Exception: pass
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if col == target_column: continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                df[col] = df[col].fillna(med if pd.notna(med) else 0)
            else:
                df[col] = df[col].fillna("Unknown").astype(str)
        except Exception: pass
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols: df[bool_cols] = df[bool_cols].astype(int)
    log.info(f"[Clean] Done. Shape: {df.shape}")
    return df


# ==============================================================================
# SECTION 6 — SCALING HELPERS
# ==============================================================================

_NEEDS_SCALE = {
    "LogisticRegression","LinearSVC","LinearSVR",
    "LinearRegression","Ridge","RidgeClassifier","Lasso","ElasticNet",
    "BayesianRidge","SGDClassifier","SGDRegressor","KNN","BernoulliNB",
}
_NEEDS_NONNEG_SCALE = {"MultinomialNB"}
_TIER1_MAX  =    10_000
_TIER2_MAX  =   100_000
_TIER3_MAX  = 1_000_000
_TIER_NAMES = {1: "SMALL", 2: "MEDIUM", 3: "LARGE", 4: "MASSIVE"}


def _get_tier(n_rows: int) -> int:
    if n_rows < _TIER1_MAX: return 1
    if n_rows < _TIER2_MAX: return 2
    if n_rows < _TIER3_MAX: return 3
    return 4

def _cap_cardinality(df, target_col, max_unique):
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col == target_col: continue
        vc = df[col].value_counts()
        if len(vc) > max_unique:
            keep = set(vc.index[:max_unique])
            df[col] = df[col].where(df[col].isin(keep), other="__other__")
    return df

def _frequency_encode(df, target_col):
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col == target_col: continue
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq).fillna(0).astype("float32")
    return df

def _downsample_for_cv(X, y, n, problem_type, random_state=42):
    if len(X) <= n: return X, y
    stratify = y if problem_type == "classification" else None
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n,
                                       random_state=random_state, stratify=stratify)
    return X_s, y_s

def _make_pipeline(name, estimator):
    if name in _NEEDS_NONNEG_SCALE:
        return Pipeline([("scaler", MinMaxScaler()), ("model", estimator)])
    if name in _NEEDS_SCALE:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator


# ==============================================================================
# SECTION 6.5 — AUTO FEATURE SELECTION
# ==============================================================================

_FS_MIN_FEATURES      = 5
_FS_CORR_THRESHOLD    = 0.95
_FS_SELECTK_TRIGGER   = 50
_FS_LDA_TRIGGER       = 20
_FS_PCA_ABS_TRIGGER   = 100
_FS_PCA_RATIO_TRIGGER = 5
_FS_PCA_VARIANCE      = 0.95
_FS_MAX_SELECTK       = 50


def _fs_step_A_variance(X_train, X_test):
    try:
        vt  = VarianceThreshold(threshold=0.0)
        vt.fit(X_train)
        mask = vt.get_support()
        kept = X_train.columns[mask].tolist()
        dropped = X_train.columns[~mask].tolist()
        Xtr = pd.DataFrame(vt.transform(X_train), columns=kept, index=X_train.index)
        Xte = pd.DataFrame(vt.transform(X_test),  columns=kept, index=X_test.index)
        return Xtr, Xte, dropped
    except Exception as exc:
        log.warning(f"[FS-A] Skipped: {exc}")
        return X_train, X_test, []

def _fs_step_B_correlation(X_train, X_test, y_train):
    try:
        try: target_corr = X_train.corrwith(y_train).abs()
        except Exception: target_corr = pd.Series(dtype=float)
        corr  = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop: continue
            partners = upper.index[upper[col] >= _FS_CORR_THRESHOLD].tolist()
            if not partners: continue
            group = [col] + [p for p in partners if p not in to_drop]
            if len(group) < 2: continue
            if len(target_corr) > 0:
                keep_col = target_corr.reindex(group).fillna(0).idxmax()
            else:
                keep_col = sorted(group)[0]
            for g in group:
                if g != keep_col: to_drop.add(g)
        to_drop = list(to_drop)
        Xtr = X_train.drop(columns=to_drop, errors="ignore") if to_drop else X_train
        Xte = X_test.drop(columns=to_drop, errors="ignore")  if to_drop else X_test
        return Xtr, Xte, to_drop
    except Exception as exc:
        log.warning(f"[FS-B] Skipped: {exc}")
        return X_train, X_test, []

def _fs_step_C_selectkbest(X_train, X_test, y_train, problem_type):
    try:
        n_feat = X_train.shape[1]; n_sample = len(X_train)
        k = max(1, min(int(np.ceil(np.sqrt(n_sample))), n_feat // 2,
                       _FS_MAX_SELECTK, n_feat))
        score_fn = f_classif if problem_type == "classification" else f_regression
        try:
            sel = SelectKBest(score_func=score_fn, k=k); sel.fit(X_train, y_train)
        except Exception:
            mi_fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
            sel = SelectKBest(score_func=mi_fn, k=k); sel.fit(X_train, y_train)
        mask = sel.get_support(); kept = X_train.columns[mask].tolist()
        scores_raw = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        scores = {c: float(s) for c, s in zip(X_train.columns, scores_raw)}
        Xtr = pd.DataFrame(sel.transform(X_train), columns=kept, index=X_train.index)
        Xte = pd.DataFrame(sel.transform(X_test),  columns=kept, index=X_test.index)
        return Xtr, Xte, kept, scores
    except Exception as exc:
        log.warning(f"[FS-C] Skipped: {exc}")
        return X_train, X_test, X_train.columns.tolist(), {}

def _fs_step_D_lda(X_train, X_test, y_train, n_classes):
    try:
        max_comp = min(X_train.shape[1], n_classes - 1)
        if max_comp < 2: return X_train, X_test, 0, False
        lda  = LinearDiscriminantAnalysis(n_components=max_comp)
        lda.fit(X_train, y_train)
        cols = [f"lda_{i}" for i in range(max_comp)]
        Xtr  = pd.DataFrame(lda.transform(X_train), columns=cols, index=X_train.index)
        Xte  = pd.DataFrame(lda.transform(X_test),  columns=cols, index=X_test.index)
        return Xtr, Xte, max_comp, True
    except Exception as exc:
        log.warning(f"[FS-D] LDA skipped: {exc}")
        return X_train, X_test, 0, False

def _fs_step_E_pca(X_train, X_test):
    try:
        n_feat = X_train.shape[1]; n_sample = len(X_train)
        max_comp = min(n_feat, n_sample - 1, 200)
        if max_comp < 2: return X_train, X_test, 0, []
        pca_probe = PCA(n_components=max_comp, random_state=42)
        pca_probe.fit(X_train)
        cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
        n_comp = max(2, min(int(np.searchsorted(cumvar, _FS_PCA_VARIANCE) + 1),
                            max_comp, 50))
        pca  = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_train)
        cols = [f"pca_{i}" for i in range(n_comp)]
        Xtr  = pd.DataFrame(pca.transform(X_train), columns=cols, index=X_train.index)
        Xte  = pd.DataFrame(pca.transform(X_test),  columns=cols, index=X_test.index)
        ev   = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        return Xtr, Xte, n_comp, ev
    except Exception as exc:
        log.warning(f"[FS-E] PCA skipped: {exc}")
        return X_train, X_test, 0, []


def auto_feature_selection(X_train, X_test, y_train, problem_type,
                            n_classes=2, tier=1):
    n_orig   = X_train.shape[1]
    n_sample = len(X_train)
    fs_report = {
        "original_features": n_orig, "final_features": n_orig,
        "steps_applied": [], "dropped_zero_var": [], "dropped_correlated": [],
        "selectkbest_kept": None, "selectkbest_top_scores": {},
        "lda_applied": False, "lda_components": 0,
        "pca_applied": False, "pca_components": 0, "pca_variance_ratios": [],
        "reduction_summary": "",
    }
    if n_orig <= _FS_MIN_FEATURES:
        fs_report["reduction_summary"] = f"Feature selection skipped: only {n_orig} feature(s)."
        return X_train, X_test, fs_report
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr, Xte, dropped_var = _fs_step_A_variance(Xtr, Xte)
    if dropped_var: fs_report["steps_applied"].append("A_variance_threshold"); fs_report["dropped_zero_var"] = dropped_var
    if Xtr.shape[1] >= 4:
        Xtr, Xte, dropped_corr = _fs_step_B_correlation(Xtr, Xte, y_train)
        if dropped_corr: fs_report["steps_applied"].append("B_correlation_filter"); fs_report["dropped_correlated"] = dropped_corr
    if tier <= 2 and Xtr.shape[1] > _FS_SELECTK_TRIGGER:
        Xtr, Xte, kept, scores = _fs_step_C_selectkbest(Xtr, Xte, y_train, problem_type)
        fs_report["steps_applied"].append("C_selectkbest")
        fs_report["selectkbest_kept"] = kept
        fs_report["selectkbest_top_scores"] = dict(
            list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:20])
    lda_applied = False
    if problem_type == "classification" and Xtr.shape[1] > _FS_LDA_TRIGGER and n_classes >= 2:
        Xtr, Xte, lda_k, lda_applied = _fs_step_D_lda(Xtr, Xte, y_train, n_classes)
        if lda_applied:
            fs_report["steps_applied"].append("D_lda")
            fs_report["lda_applied"] = True; fs_report["lda_components"] = lda_k
    pca_trigger_abs   = Xtr.shape[1] > _FS_PCA_ABS_TRIGGER
    pca_trigger_ratio = (Xtr.shape[1] > 30 and n_sample < Xtr.shape[1] * _FS_PCA_RATIO_TRIGGER)
    if not lda_applied and (pca_trigger_abs or pca_trigger_ratio):
        Xtr, Xte, pca_k, ev = _fs_step_E_pca(Xtr, Xte)
        if pca_k > 0:
            fs_report["steps_applied"].append("E_pca")
            fs_report["pca_applied"] = True; fs_report["pca_components"] = pca_k
            fs_report["pca_variance_ratios"] = ev
    n_final = Xtr.shape[1]
    fs_report["final_features"] = n_final
    if fs_report["steps_applied"]:
        pct = 100 * (1 - n_final / max(n_orig, 1))
        fs_report["reduction_summary"] = (
            f"Feature reduction: {n_orig} → {n_final} features ({pct:.1f}% reduction). "
            f"Steps: {', '.join(fs_report['steps_applied'])}.")
    else:
        fs_report["reduction_summary"] = f"No reduction — all {n_orig} features retained."
    return Xtr, Xte, fs_report


# ==============================================================================
# SECTION 7 — MODEL QUALITY COMMENTARY
# ==============================================================================

def _quality_commentary(problem_type, accuracy, r2, mae, baseline_acc, n_classes):
    result = {"rating": "unknown", "summary": "", "details": "", "baseline_accuracy": baseline_acc}
    if problem_type == "classification" and accuracy is not None:
        bl  = baseline_acc or (1.0 / max(n_classes, 1))
        gap = accuracy - bl; pct = accuracy * 100
        if accuracy >= 0.90 and gap >= 0.20:
            rating  = "Excellent"; summary = f"Model accuracy {pct:.1f}% is excellent."
        elif accuracy >= 0.75 and gap >= 0.10:
            rating  = "Good"; summary = f"Model accuracy {pct:.1f}% is good."
        elif gap > 0.02:
            rating  = "Fair"; summary = f"Model accuracy {pct:.1f}% is fair."
        else:
            rating  = "Poor"; summary = f"Accuracy at or near baseline ({bl*100:.1f}%)."
        result.update({"rating": rating, "summary": summary,
                       "details": f"For {n_classes}-class, 75-90% good, >90% excellent."})
    elif problem_type == "regression" and r2 is not None:
        if r2 >= 0.90:   rating = "Excellent"; summary = f"R²={r2:.4f} — excellent fit."
        elif r2 >= 0.75: rating = "Good";      summary = f"R²={r2:.4f} — good fit."
        elif r2 >= 0.50: rating = "Fair";      summary = f"R²={r2:.4f} — fair fit."
        else:            rating = "Poor";      summary = f"R²={r2:.4f} — poor fit."
        mae_str = f"  MAE={mae:.4f}." if mae is not None else ""
        result.update({"rating": rating, "summary": summary,
                       "details": f"R² closer to 1.0 is better.{mae_str}"})
    return result


# ==============================================================================
# SECTION 8 — ALGORITHM REGISTRY (unchanged — omitted for brevity)
# ==============================================================================

CLASSIFICATION_ALGORITHMS = {
    "LogisticRegression":           {"category":"Linear","task":"classification","scales_to":"Tier 1–3","supports_proba":True,"supports_multiclass":True,"description":"Probabilistic linear classifier.","variants":["lbfgs","liblinear","saga"]},
    "RidgeClassifier":              {"category":"Linear","task":"classification","scales_to":"Tier 1–3","supports_proba":False,"supports_multiclass":True,"description":"Ridge-regularised linear classifier.","variants":["standard Ridge (L2)"]},
    "SGDClassifier":                {"category":"Linear / SGD","task":"classification","scales_to":"Tier 1–4","supports_proba":True,"supports_multiclass":True,"description":"Stochastic Gradient Descent.","variants":["hinge","modified_huber","log_loss"]},
    "LinearSVC":                    {"category":"SVM","task":"classification","scales_to":"Tier 1–3","supports_proba":False,"supports_multiclass":True,"description":"Linear-kernel SVM.","variants":["L1 penalty","L2 penalty"]},
    "KNeighborsClassifier":         {"category":"Instance-Based","task":"classification","scales_to":"Tier 1","supports_proba":True,"supports_multiclass":True,"description":"K-Nearest Neighbours.","variants":["uniform","distance"]},
    "DecisionTreeClassifier":       {"category":"Tree","task":"classification","scales_to":"Tier 1–2","supports_proba":True,"supports_multiclass":True,"description":"Single decision tree.","variants":["gini","entropy"]},
    "RandomForestClassifier":       {"category":"Ensemble – Bagging","task":"classification","scales_to":"Tier 1–2","supports_proba":True,"supports_multiclass":True,"description":"Bootstrap-aggregated ensemble.","variants":["sqrt","log2"]},
    "ExtraTreesClassifier":         {"category":"Ensemble – Bagging","task":"classification","scales_to":"Tier 1–2","supports_proba":True,"supports_multiclass":True,"description":"Extremely Randomised Trees.","variants":["sqrt","log2"]},
    "GradientBoostingClassifier":   {"category":"Ensemble – Boosting","task":"classification","scales_to":"Tier 1","supports_proba":True,"supports_multiclass":True,"description":"sklearn sequential boosting.","variants":["deviance","exponential"]},
    "HistGradientBoostingClassifier":{"category":"Ensemble – Boosting","task":"classification","scales_to":"Tier 1–3","supports_proba":True,"supports_multiclass":True,"description":"Histogram-based gradient boosting.","variants":["log_loss"]},
    "XGBClassifier":                {"category":"Ensemble – Boosting","task":"classification","scales_to":"Tier 1–3","supports_proba":True,"supports_multiclass":True,"description":"eXtreme Gradient Boosting.","variants":["gbtree","dart"]},
    "LGBMClassifier":               {"category":"Ensemble – Boosting","task":"classification","scales_to":"Tier 1–4","supports_proba":True,"supports_multiclass":True,"description":"LightGBM.","variants":["gbdt","goss","dart"]},
    "CatBoostClassifier":           {"category":"Ensemble – Boosting","task":"classification","scales_to":"Tier 1–3","supports_proba":True,"supports_multiclass":True,"description":"CatBoost.","variants":["Ordered","Plain"]},
    "GaussianNB":                   {"category":"Probabilistic","task":"classification","scales_to":"Tier 1–4","supports_proba":True,"supports_multiclass":True,"description":"Naïve Bayes Gaussian.","variants":["var_smoothing"]},
    "BernoulliNB":                  {"category":"Probabilistic","task":"classification","scales_to":"Tier 1–4","supports_proba":True,"supports_multiclass":True,"description":"Naïve Bayes binary.","variants":["binarize threshold"]},
    "MultinomialNB":                {"category":"Probabilistic","task":"classification","scales_to":"Tier 1–3","supports_proba":True,"supports_multiclass":True,"description":"Naïve Bayes multinomial.","variants":["alpha"]},
}

REGRESSION_ALGORITHMS = {
    "LinearRegression":             {"category":"Linear","task":"regression","scales_to":"Tier 1–3","description":"OLS.","variants":["OLS"]},
    "Ridge":                        {"category":"Linear – Regularised","task":"regression","scales_to":"Tier 1–4","description":"L2.","variants":["alpha"]},
    "Lasso":                        {"category":"Linear – Regularised","task":"regression","scales_to":"Tier 1–2","description":"L1.","variants":["alpha"]},
    "ElasticNet":                   {"category":"Linear – Regularised","task":"regression","scales_to":"Tier 1","description":"L1+L2.","variants":["l1_ratio"]},
    "BayesianRidge":                {"category":"Probabilistic","task":"regression","scales_to":"Tier 1–2","description":"Bayesian linear.","variants":["n_iter"]},
    "SGDRegressor":                 {"category":"Linear / SGD","task":"regression","scales_to":"Tier 1–4","description":"SGD regressor.","variants":["squared_error","huber"]},
    "LinearSVR":                    {"category":"SVM","task":"regression","scales_to":"Tier 1–3","description":"Linear SVR.","variants":["epsilon"]},
    "KNeighborsRegressor":          {"category":"Instance-Based","task":"regression","scales_to":"Tier 1","description":"K-NN regressor.","variants":["k"]},
    "DecisionTreeRegressor":        {"category":"Tree","task":"regression","scales_to":"Tier 1–2","description":"Single tree.","variants":["mse","mae"]},
    "RandomForestRegressor":        {"category":"Ensemble – Bagging","task":"regression","scales_to":"Tier 1–2","description":"Bootstrap trees.","variants":["sqrt","log2"]},
    "ExtraTreesRegressor":          {"category":"Ensemble – Bagging","task":"regression","scales_to":"Tier 1–2","description":"Extra Trees.","variants":["sqrt","log2"]},
    "GradientBoostingRegressor":    {"category":"Ensemble – Boosting","task":"regression","scales_to":"Tier 1","description":"Sequential boosting.","variants":["ls","huber"]},
    "HistGradientBoostingRegressor":{"category":"Ensemble – Boosting","task":"regression","scales_to":"Tier 1–3","description":"Histogram GBR.","variants":["squared_error"]},
    "XGBRegressor":                 {"category":"Ensemble – Boosting","task":"regression","scales_to":"Tier 1–3","description":"XGBoost regressor.","variants":["gbtree"]},
    "LGBMRegressor":                {"category":"Ensemble – Boosting","task":"regression","scales_to":"Tier 1–4","description":"LightGBM regressor.","variants":["gbdt"]},
    "CatBoostRegressor":            {"category":"Ensemble – Boosting","task":"regression","scales_to":"Tier 1–3","description":"CatBoost regressor.","variants":["RMSE"]},
}

ALGORITHM_SUMMARY = {
    "total_algorithms": len(CLASSIFICATION_ALGORITHMS) + len(REGRESSION_ALGORITHMS),
    "classification_count": len(CLASSIFICATION_ALGORITHMS),
    "regression_count": len(REGRESSION_ALGORITHMS),
}

def get_algorithm_info(name):
    return (CLASSIFICATION_ALGORITHMS.get(name) or REGRESSION_ALGORITHMS.get(name) or {})

def algorithm_summary_for_api():
    return {"summary": ALGORITHM_SUMMARY,
            "classification": CLASSIFICATION_ALGORITHMS,
            "regression": REGRESSION_ALGORITHMS}


# ==============================================================================
# SECTION 9 — HYPERPARAMETER SEARCH GRIDS
# ==============================================================================

_CV_GRIDS_CLS = {
    "RandomForest":         {"n_estimators":[100,200,300],"max_features":["sqrt","log2"],"min_samples_leaf":[1,2,4]},
    "ExtraTrees":           {"n_estimators":[100,300],"max_features":["sqrt","log2"],"min_samples_leaf":[1,2]},
    "GradientBoosting":     {"n_estimators":[50,100],"learning_rate":[0.05,0.1,0.2],"max_depth":[3,4,5]},
    "HistGradientBoosting": {"max_iter":[100,200],"learning_rate":[0.05,0.1,0.2],"max_depth":[None,5,10]},
    "XGBoost":              {"n_estimators":[100,200,300],"learning_rate":[0.05,0.1,0.2],"max_depth":[3,4,5,6],"subsample":[0.8,1.0]},
    "LightGBM":             {"n_estimators":[100,200,300],"learning_rate":[0.05,0.1,0.2],"num_leaves":[15,31,63],"min_child_samples":[10,20]},
    "CatBoost":             {"iterations":[100,300],"learning_rate":[0.05,0.1],"depth":[4,6]},
    "LogisticRegression":   {"C":[0.01,0.1,1.0,10.0],"solver":["lbfgs","saga"]},
    "LinearSVC":            {"C":[0.01,0.1,1.0,10.0],"max_iter":[1000,2000]},
    "SGDClassifier":        {"loss":["modified_huber","log_loss"],"alpha":[0.0001,0.001],"penalty":["l2","l1"]},
    "AdaBoost":             {"n_estimators":[50,100,200],"learning_rate":[0.5,1.0,2.0]},
    "Bagging":              {"n_estimators":[10,20,50],"max_samples":[0.7,0.9,1.0]},
    "KNN":                  {"n_neighbors":[3,5,7,10,15],"weights":["uniform","distance"]},
    "DecisionTree":         {"max_depth":[3,5,8,10],"min_samples_leaf":[2,4,8],"criterion":["gini","entropy"],"ccp_alpha":[0.0,0.001]},
    "GaussianNB":           {"var_smoothing":[1e-9,1e-8,1e-7,1e-6]},
    "BernoulliNB":          {"alpha":[0.1,0.5,1.0,2.0]},
    "MultinomialNB":        {"alpha":[0.1,0.5,1.0,2.0]},
    "QDA":                  {"reg_param":[0.0,0.1,0.5]},
}

_CV_GRIDS_REG = {
    "LinearRegression":     {},
    "Ridge":                {"alpha":[0.001,0.01,0.1,1.0,10.0,100.0]},
    "Lasso":                {"alpha":[0.001,0.01,0.1,1.0],"max_iter":[5000,10000]},
    "ElasticNet":           {"alpha":[0.01,0.1,1.0],"l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
    "BayesianRidge":        {"n_iter":[100,300]},
    "LinearSVR":            {"C":[0.1,1.0,10.0],"epsilon":[0.0,0.1,0.5],"max_iter":[2000,5000]},
    "SGDRegressor":         {"loss":["squared_error","huber"],"alpha":[0.0001,0.001],"penalty":["l2","l1"]},
    "KNN":                  {"n_neighbors":[3,5,7,10],"weights":["uniform","distance"]},
    "DecisionTree":         {"max_depth":[3,5,8,10],"min_samples_leaf":[2,4,8],"ccp_alpha":[0.0,0.001]},
    "RandomForest":         {"n_estimators":[100,200,300],"max_depth":[None,5,10,20],"min_samples_leaf":[1,2,4]},
    "ExtraTrees":           {"n_estimators":[100,200],"max_depth":[None,5,10],"min_samples_leaf":[1,2]},
    "GradientBoosting":     {"n_estimators":[50,100],"learning_rate":[0.05,0.1,0.2],"max_depth":[3,4,5]},
    "HistGradientBoosting": {"max_iter":[100,200],"learning_rate":[0.05,0.1,0.2],"max_depth":[None,5,10]},
    "XGBoost":              {"n_estimators":[100,200,300],"learning_rate":[0.05,0.1,0.2],"max_depth":[3,4,5,6],"subsample":[0.8,1.0]},
    "LightGBM":             {"n_estimators":[100,200,300],"learning_rate":[0.05,0.1,0.2],"num_leaves":[15,31,63]},
    "CatBoost":             {"iterations":[100,300],"learning_rate":[0.05,0.1],"depth":[4,6]},
}


def _search_hyperparams(name, estimator, param_grid, X_cv, y_cv,
                        cv_splitter, scoring, tier):
    pipe = _make_pipeline(name, estimator)
    if isinstance(pipe, Pipeline) and param_grid:
        pg = {f"model__{k}": v for k, v in param_grid.items()}
    else:
        pg = dict(param_grid)
    if not pg:
        scores = cross_val_score(pipe, X_cv, y_cv, cv=cv_splitter,
                                 scoring=scoring, n_jobs=-1)
        return float(scores.mean()), float(scores.std()), {}
    n_combos = 1
    for v in pg.values(): n_combos *= len(v) if hasattr(v, "__len__") else 1
    use_grid = (tier <= 2 and n_combos <= 30)
    n_iter   = min(20, n_combos)
    if use_grid:
        search = GridSearchCV(pipe, pg, cv=cv_splitter, scoring=scoring,
                              n_jobs=-1, refit=False, error_score=np.nan)
    else:
        search = RandomizedSearchCV(pipe, pg, n_iter=n_iter, cv=cv_splitter,
                                    scoring=scoring, n_jobs=-1, refit=False,
                                    random_state=42, error_score=np.nan)
    search.fit(X_cv, y_cv)
    best_idx = search.best_index_
    best_std = float(np.nan_to_num(search.cv_results_["std_test_score"][best_idx]))
    return float(search.best_score_), best_std, search.best_params_


def _build_final_model(name, base_estimator, best_params):
    final = clone(base_estimator)
    clean = {k.replace("model__", ""): v for k, v in best_params.items()}
    if "n_estimators" in clean:
        clean["n_estimators"] = min(int(clean["n_estimators"] * 2), 600)
    if "max_iter" in clean and name in ("HistGradientBoosting","LightGBM",
                                         "SGDClassifier","SGDRegressor"):
        clean["max_iter"] = min(int(clean["max_iter"] * 2), 600)
    if "iterations" in clean:
        clean["iterations"] = min(int(clean["iterations"] * 2), 600)
    if clean: final.set_params(**clean)
    return _make_pipeline(name, final)


# ==============================================================================
# SECTION 10 — CANDIDATE BUILDERS
# ==============================================================================

def _apply_small_dataset_constraints(param_grid):
    capped = dict(param_grid)
    if "max_depth" in capped:
        capped["max_depth"] = [d for d in capped["max_depth"]
                               if d is None or (isinstance(d, int) and d <= 5)]
        if not capped["max_depth"]: capped["max_depth"] = [3, 5]
    if "n_estimators" in capped:
        capped["n_estimators"] = [n for n in capped["n_estimators"] if n <= 100]
        if not capped["n_estimators"]: capped["n_estimators"] = [50, 100]
    return capped


def _build_cls_candidates(tier, small_dataset=False):
    if tier == 4:
        c = {"SGDClassifier": (
            SGDClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            _CV_GRIDS_CLS["SGDClassifier"])}
        if _LGBM_AVAILABLE and not small_dataset:
            c["LightGBM"] = (LGBMClassifier(verbose=-1, random_state=42, n_jobs=-1,
                                             class_weight="balanced"), _CV_GRIDS_CLS["LightGBM"])
        return c
    c = {}
    c["LogisticRegression"] = (LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), _CV_GRIDS_CLS["LogisticRegression"])
    c["LinearSVC"]          = (LinearSVC(random_state=42, class_weight="balanced"), _CV_GRIDS_CLS["LinearSVC"])
    if tier <= 2:
        c["RandomForest"] = (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                             _apply_small_dataset_constraints(_CV_GRIDS_CLS["RandomForest"]) if small_dataset else _CV_GRIDS_CLS["RandomForest"])
        c["ExtraTrees"]   = (ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                             _apply_small_dataset_constraints(_CV_GRIDS_CLS["ExtraTrees"]) if small_dataset else _CV_GRIDS_CLS["ExtraTrees"])
    if tier == 1:
        c["GradientBoosting"]     = (GradientBoostingClassifier(random_state=42), _apply_small_dataset_constraints(_CV_GRIDS_CLS["GradientBoosting"]) if small_dataset else _CV_GRIDS_CLS["GradientBoosting"])
        c["HistGradientBoosting"] = (HistGradientBoostingClassifier(random_state=42, class_weight="balanced"), _CV_GRIDS_CLS["HistGradientBoosting"])
        c["AdaBoost"]      = (AdaBoostClassifier(random_state=42),    _CV_GRIDS_CLS["AdaBoost"])
        c["Bagging"]       = (BaggingClassifier(random_state=42, n_jobs=-1), _CV_GRIDS_CLS["Bagging"])
        c["KNN"]           = (KNeighborsClassifier(),                  _CV_GRIDS_CLS["KNN"])
        c["DecisionTree"]  = (DecisionTreeClassifier(random_state=42, class_weight="balanced"), _apply_small_dataset_constraints(_CV_GRIDS_CLS["DecisionTree"]) if small_dataset else _CV_GRIDS_CLS["DecisionTree"])
        c["GaussianNB"]    = (GaussianNB(),    _CV_GRIDS_CLS["GaussianNB"])
        c["BernoulliNB"]   = (BernoulliNB(),   _CV_GRIDS_CLS["BernoulliNB"])
        c["MultinomialNB"] = (MultinomialNB(), _CV_GRIDS_CLS["MultinomialNB"])
        c["QDA"]           = (QuadraticDiscriminantAnalysis(), _CV_GRIDS_CLS["QDA"])
    if tier == 2:
        c["HistGradientBoosting"] = (HistGradientBoostingClassifier(random_state=42, class_weight="balanced"), _CV_GRIDS_CLS["HistGradientBoosting"])
    if not small_dataset:
        if _XGB_AVAILABLE and tier <= 3:
            c["XGBoost"] = (XGBClassifier(eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=-1), _CV_GRIDS_CLS["XGBoost"])
        if _LGBM_AVAILABLE:
            c["LightGBM"] = (LGBMClassifier(verbose=-1, random_state=42, n_jobs=-1, class_weight="balanced"), _CV_GRIDS_CLS["LightGBM"])
        if _CATBOOST_AVAILABLE and tier <= 3:
            c["CatBoost"] = (CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced"), _CV_GRIDS_CLS["CatBoost"])
    return c


def _build_reg_candidates(tier, low_feature_mode=False, small_dataset=False):
    if tier == 4:
        c = {"SGDRegressor": (SGDRegressor(random_state=42), _CV_GRIDS_REG["SGDRegressor"])}
        if _LGBM_AVAILABLE and not small_dataset:
            c["LightGBM"] = (LGBMRegressor(verbose=-1, random_state=42, n_jobs=-1), _CV_GRIDS_REG["LightGBM"])
        return c
    c = {}
    c["LinearRegression"] = (LinearRegression(), _CV_GRIDS_REG["LinearRegression"])
    c["Ridge"]            = (Ridge(), _CV_GRIDS_REG["Ridge"])
    c["LinearSVR"]        = (LinearSVR(max_iter=2000), _CV_GRIDS_REG["LinearSVR"])
    if low_feature_mode:
        c["Lasso"]         = (Lasso(max_iter=5000), _CV_GRIDS_REG["Lasso"])
        c["ElasticNet"]    = (ElasticNet(max_iter=5000), _CV_GRIDS_REG["ElasticNet"])
        c["BayesianRidge"] = (BayesianRidge(), _CV_GRIDS_REG["BayesianRidge"])
    if tier <= 2:
        c["RandomForest"] = (RandomForestRegressor(random_state=42, n_jobs=-1), _apply_small_dataset_constraints(_CV_GRIDS_REG["RandomForest"]) if small_dataset else _CV_GRIDS_REG["RandomForest"])
        c["ExtraTrees"]   = (ExtraTreesRegressor(random_state=42, n_jobs=-1),   _apply_small_dataset_constraints(_CV_GRIDS_REG["ExtraTrees"])   if small_dataset else _CV_GRIDS_REG["ExtraTrees"])
    if tier == 1:
        c["GradientBoosting"]     = (GradientBoostingRegressor(random_state=42), _apply_small_dataset_constraints(_CV_GRIDS_REG["GradientBoosting"]) if small_dataset else _CV_GRIDS_REG["GradientBoosting"])
        c["HistGradientBoosting"] = (HistGradientBoostingRegressor(random_state=42), _CV_GRIDS_REG["HistGradientBoosting"])
        c["KNN"]                  = (KNeighborsRegressor(), _CV_GRIDS_REG["KNN"])
        c["DecisionTree"]         = (DecisionTreeRegressor(random_state=42), _apply_small_dataset_constraints(_CV_GRIDS_REG["DecisionTree"]) if small_dataset else _CV_GRIDS_REG["DecisionTree"])
    if tier == 2:
        c["HistGradientBoosting"] = (HistGradientBoostingRegressor(random_state=42), _CV_GRIDS_REG["HistGradientBoosting"])
    if not small_dataset:
        if _XGB_AVAILABLE and tier <= 3: c["XGBoost"]  = (XGBRegressor(verbosity=0, random_state=42, n_jobs=-1), _CV_GRIDS_REG["XGBoost"])
        if _LGBM_AVAILABLE:              c["LightGBM"] = (LGBMRegressor(verbose=-1, random_state=42, n_jobs=-1), _CV_GRIDS_REG["LightGBM"])
        if _CATBOOST_AVAILABLE and tier <= 3: c["CatBoost"] = (CatBoostRegressor(verbose=0, random_state=42), _CV_GRIDS_REG["CatBoost"])
    return c


# ==============================================================================
# SECTION 10.5 — META-LEARNING PRIORITISATION  (v5.6 — model-first)
# ==============================================================================

def _apply_meta_learning_priority(candidates: dict, n_rows: int,
                                   n_features: int, problem_type: str,
                                   missing_pct: float = 0.0) -> dict:
    """
    v5.6 — Model-first candidate prioritisation.

    Attempts MetaPriorityScorer (learned from cross-run outcome ledger) first.
    Falls back to the minimal rule-based heuristic only when:
      - scorer has fewer than MIN_SAMPLES records, OR
      - sklearn is unavailable

    Rules below are a SAFETY NET for cold-boot only.
    """
    # Re-warm scorer in case new ledger data arrived
    try:
        import json as _json
        lp = Path("agent_system_state") / "outcome_ledger.json"
        if lp.exists():
            with open(lp) as fh:
                records = _json.load(fh).get("records", [])
            _global_priority_scorer.maybe_fit(records)
    except Exception:
        pass

    priority_scores: dict = {}

    # ── Primary: learned scorer ───────────────────────────────────────────────
    any_scored = False
    for name in candidates:
        learned = _global_priority_scorer.score(
            name, float(n_rows), float(n_features), problem_type, missing_pct)
        if learned is not None:
            priority_scores[name] = learned
            any_scored = True

    if any_scored:
        ordered = dict(sorted(candidates.items(),
                              key=lambda x: priority_scores.get(x[0], 0.0),
                              reverse=True))
        log.info(f"[MetaLearn] Learned priority order: {list(ordered.keys())[:5]}…")
        return ordered

    # ── Safety fallback — rule heuristic (cold-boot only) ────────────────────
    linear_names   = {"LogisticRegression","LinearSVC","LinearSVR","Ridge",
                      "Lasso","ElasticNet","BayesianRidge","RidgeClassifier"}
    nb_names       = {"GaussianNB","BernoulliNB","MultinomialNB"}
    boosting_names = {"XGBoost","LightGBM","CatBoost","GradientBoosting","HistGradientBoosting"}
    tree_names     = {"RandomForest","ExtraTrees","DecisionTree"}
    for name in candidates:
        score = 0.0
        if n_rows < 500:
            if name in linear_names:   score += 2.0
            if name in nb_names:       score += 1.5
            if name in boosting_names: score -= 1.0
        if n_rows >= 10_000:
            if name in boosting_names: score += 2.0
            if name in tree_names:     score += 1.0
        if n_features > n_rows:
            if name in linear_names:   score += 1.5
            if name in boosting_names: score -= 0.5
        if n_features <= 10:
            if name in tree_names:     score += 1.0
            if name in boosting_names: score += 0.5
        priority_scores[name] = score
    ordered = dict(sorted(candidates.items(),
                          key=lambda x: priority_scores.get(x[0], 0), reverse=True))
    log.info(f"[MetaLearn] Rule-fallback priority: {list(ordered.keys())[:5]}…")
    return ordered


# ==============================================================================
# SECTION 10.8 — STACKING ENSEMBLE
# ==============================================================================

def _build_stacking_model(best_scores, candidates, best_params_all,
                           problem_type, top_n=3):
    if len(best_scores) < 2:
        return None, []
    sorted_models = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    top_names     = [n for n, _ in sorted_models[:top_n]]
    estimators    = []
    for name in top_names:
        if name not in candidates: continue
        base_est, _ = candidates[name]
        final_est   = _build_final_model(name, base_est, best_params_all.get(name, {}))
        estimators.append((name, final_est))
    if len(estimators) < 2: return None, []
    try:
        if problem_type == "classification":
            meta  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            stack = StackingClassifier(estimators=estimators, final_estimator=meta,
                                       cv=3, n_jobs=-1, passthrough=False)
        else:
            meta  = Ridge(alpha=1.0)
            stack = StackingRegressor(estimators=estimators, final_estimator=meta,
                                      cv=3, n_jobs=-1, passthrough=False)
        return stack, top_names
    except Exception as exc:
        log.warning(f"[Stacking] Failed: {exc}")
        return None, []


# ==============================================================================
# SECTION 11 — FEATURE IMPORTANCE, SHAP, CHART, SAVE, PDF
# ==============================================================================

_REPORT_DIR = Path("app/reports")
_MODEL_DIR  = Path("models")
_PDF_DIR    = Path("app/reports")


def _load_file(file_path):
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            with open(file_path, "rb") as fh:
                raw = fh.read().replace(b"\x00", b"")
            return pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding_errors="replace")
        if suffix == ".xlsx": return pd.read_excel(file_path, engine="openpyxl")
        if suffix == ".xls":  return pd.read_excel(file_path, engine="xlrd")
        if suffix == ".json": return pd.read_json(file_path)
        if suffix == ".parquet": return pd.read_parquet(file_path)
        if suffix == ".txt":
            return pd.read_csv(file_path, sep=None, engine="python", encoding_errors="replace")
        return {"error": f"Unsupported: '{suffix}'"}
    except Exception as exc:
        return {"error": f"File reading error: {exc}"}


def _detect_problem_type(y, n_feats):
    y_num       = pd.to_numeric(y, errors="coerce")
    n_rows      = len(y)
    if y_num.isna().any(): return "classification", "non-numeric target", y
    n_unique     = int(y_num.nunique())
    unique_ratio = n_unique / max(n_rows, 1)
    has_fractions = bool((y_num % 1 != 0).any())
    if n_unique <= 2:   return "classification", "binary", y_num
    if has_fractions:   return "regression",     "decimal values", y_num
    if unique_ratio > 0.05: return "regression", f"ratio={unique_ratio:.3f}", y_num
    if n_unique <= 15:  return "classification", "multi-class", y_num
    return "regression", f"n_unique={n_unique}", y_num


def _engineer_low_features(X_train, X_test, y_train):
    note = ""
    X_train_r = X_train.copy(); X_test_r = X_test.copy()
    orig_cols = X_train.columns.tolist()
    try:
        poly    = PolynomialFeatures(degree=2, include_bias=False)
        tr_poly = poly.fit_transform(X_train_r)
        te_poly = poly.transform(X_test_r)
        poly_cols   = poly.get_feature_names_out(orig_cols)
        X_train_r   = pd.DataFrame(tr_poly, columns=poly_cols, index=X_train.index)
        X_test_r    = pd.DataFrame(te_poly, columns=poly_cols, index=X_test.index)
        non_const   = X_train_r.columns[X_train_r.std() > 0]
        X_train_r   = X_train_r[non_const]; X_test_r = X_test_r[non_const]
        note = f"Polynomial terms added ({len(orig_cols)} → {X_train_r.shape[1]})."
    except Exception as exc:
        log.warning(f"[LowFeat] Polynomial step failed: {exc}")
    for i, c1 in enumerate(orig_cols):
        for c2 in orig_cols[i + 1:]:
            try:
                rname = f"{c1}_div_{c2}"
                if rname not in X_train_r.columns:
                    X_train_r[rname] = (X_train[c1] / X_train[c2].replace(0, np.nan)).fillna(0)
                    X_test_r[rname]  = (X_test[c1]  / X_test[c2].replace(0, np.nan)).fillna(0)
            except Exception: pass
    return X_train_r, X_test_r, note


def _get_feature_importance(model, columns, fs_report=None):
    inner = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(inner, "feature_importances_"):
        return {c: float(_safe_float(v) or 0.0) for c, v in zip(columns, inner.feature_importances_)}
    if hasattr(inner, "coef_"):
        coef = inner.coef_
        if coef.ndim > 1: coef = np.abs(coef).mean(axis=0)
        return {c: float(abs(_safe_float(v) or 0.0)) for c, v in zip(columns, coef)}
    if fs_report and fs_report.get("pca_applied"):
        ev = fs_report.get("pca_variance_ratios", [])
        if ev: return {c: float(ev[i]) if i < len(ev) else 0.0 for i, c in enumerate(columns)}
    return {c: 0.0 for c in columns}


def _make_feature_chart(feat_imp, feature_correlations, best_name, problem_type):
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"feature_{uuid.uuid4().hex}.png"
    try:
        top_n   = min(10, len(feat_imp))
        top_imp = dict(sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
        has_corr = problem_type == "regression" and bool(feature_correlations)
        chart_h  = max(3, min(8, top_n * 0.6 + 1.5))
        fig, axes = plt.subplots(1, 2 if has_corr else 1,
                                 figsize=(14 if has_corr else 10, chart_h))
        ax_imp = axes[0] if isinstance(axes, np.ndarray) else axes
        sns.barplot(x=list(top_imp.values()), y=list(top_imp.keys()), ax=ax_imp, palette="Blues_r")
        ax_imp.set_title(f"Feature Importances — {best_name}")
        if has_corr and isinstance(axes, np.ndarray):
            ax_cor  = axes[1]
            cor_items = sorted(feature_correlations.items(), key=lambda kv: abs(kv[1] or 0), reverse=True)[:top_n]
            colors    = ["#e74c3c" if v < 0 else "#2ecc71" for _, v in cor_items]
            ax_cor.barh([k for k, _ in cor_items], [v or 0.0 for _, v in cor_items], color=colors)
            ax_cor.set_title("Feature ↔ Target Correlation")
        plt.tight_layout(); plt.savefig(_REPORT_DIR / fname, dpi=100); plt.close(fig)
    except Exception as exc:
        log.warning(f"[Chart] Failed: {exc}"); fname = ""
    return fname


def _get_shap_explanation(model, X_train, X_test, shap_sample_n=None):
    cols = X_test.columns.tolist()
    is_latent = (all(c.startswith("pca_") for c in cols) or
                 all(c.startswith("lda_") for c in cols))
    if is_latent:
        imp = _get_feature_importance(model, cols)
        return {"base_value": 0.0, "feature_values": {c: 0.0 for c in cols},
                "shap_values": imp, "available": True, "note": "PCA/LDA applied."}
    try:
        if shap_sample_n is not None and len(X_test) > shap_sample_n:
            X_test = X_test.sample(n=shap_sample_n, random_state=42)
        if isinstance(model, Pipeline):
            scaler      = model.named_steps["scaler"]
            inner_model = model.named_steps["model"]
            X_tr = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_te = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns,  index=X_test.index)
        else:
            inner_model = model; X_tr, X_te = X_train, X_test
        if hasattr(inner_model, "feature_importances_"):
            explainer = shap.TreeExplainer(inner_model); shap_values = explainer.shap_values(X_te); base_val = explainer.expected_value
        elif hasattr(inner_model, "coef_"):
            explainer = shap.LinearExplainer(inner_model, X_tr, feature_perturbation="correlation_dependent"); shap_values = explainer.shap_values(X_te); base_val = explainer.expected_value
        else:
            background = shap.sample(X_tr, min(50, len(X_tr)))
            predict_fn = (inner_model.predict_proba if hasattr(inner_model, "predict_proba") else inner_model.predict)
            explainer  = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_te.iloc[:1]); base_val = explainer.expected_value; X_te = X_te.iloc[:1]
        if isinstance(shap_values, list): sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: sv = np.mean(np.abs(shap_values), axis=0)
        else: sv = shap_values
        sv = np.atleast_2d(sv); sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0); sv_row = sv[0]
        bv = float(base_val[0] if isinstance(base_val, (list, np.ndarray)) else base_val)
        return {"base_value": bv, "feature_values": {c: float(v) for c, v in zip(X_te.columns, X_te.iloc[0].values)},
                "shap_values": {c: float(v) for c, v in zip(X_te.columns, sv_row)}, "available": True}
    except Exception as exc:
        log.warning(f"[SHAP] Not available: {exc}")
        return {"error": str(exc), "base_value": 0.0, "feature_values": {}, "shap_values": {}, "available": False}


def _save_model(model, columns, problem_type, label_encoder=None):
    _MODEL_DIR.mkdir(exist_ok=True)
    fname = f"model_{uuid.uuid4().hex}.pkl"
    try:
        payload = {"model": model, "columns": columns, "problem_type": problem_type}
        if label_encoder is not None: payload["label_encoder"] = label_encoder
        joblib.dump(payload, _MODEL_DIR / fname)
    except Exception as exc:
        log.warning(f"[Save] Failed: {exc}"); fname = ""
    return fname


def _save_pdf(best_name, problem_type, model_results, perf_report, chart_fname, quality):
    _PDF_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"report_{uuid.uuid4().hex}.pdf"
    try:
        c = rl_canvas.Canvas(str(_PDF_DIR / fname), pagesize=letter)
        w, h = letter
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h - 40, "AutoML Report — v5.6 Learning AI Edition")
        fs      = perf_report.get("feature_selection", {})
        lines   = [
            f"Problem Type  : {problem_type}",
            f"Best Model    : {best_name}",
            f"Confidence    : {perf_report.get('confidence_label','N/A')} ({perf_report.get('confidence_score','N/A')})",
            f"Feature Sel.  : {fs.get('reduction_summary', 'N/A')}",
            f"Overfitting   : {perf_report.get('overfitting', 'N/A')}",
        ]
        if problem_type == "classification":
            lines += [f"CV Score  : {perf_report.get('cv_score_mean','N/A')} ±{perf_report.get('cv_score_std','N/A')}",
                      f"Accuracy  : {perf_report.get('accuracy','N/A')}",
                      f"Quality   : {quality.get('rating','?')} — {quality.get('summary','')}"]
        else:
            lines += [f"CV R²    : {perf_report.get('cv_r2_mean','N/A')} ±{perf_report.get('cv_r2_std','N/A')}",
                      f"Test R²  : {perf_report.get('R2','N/A')}",
                      f"Quality  : {quality.get('rating','?')} — {quality.get('summary','')}"]
        c.setFont("Helvetica", 10)
        txt = c.beginText(40, h - 75)
        for line in lines: txt.textLine(str(line))
        c.drawText(txt)
        chart_path = _REPORT_DIR / chart_fname
        if chart_fname and chart_path.exists():
            c.drawImage(str(chart_path), 40, h - 500, width=500, height=280)
        c.save()
    except Exception as exc:
        log.warning(f"[PDF] Failed: {exc}"); fname = ""
    return fname


# ==============================================================================
# SECTION 12 — MAIN AutoML FUNCTION  run_automl  v5.6
# ==============================================================================

def load_data(filename):
    file_path = Path("uploads") / filename
    if not file_path.exists(): return {"error": f"File not found: {file_path}"}
    return _load_file(file_path)

def preprocess_data(df, target_column):
    return universal_cleaning(df, target_column)

def split_data(X, y, problem_type, test_size=0.20):
    stratify = y if problem_type == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=42,
                            shuffle=True, stratify=stratify)

def train_model(name, estimator, X_train, y_train):
    pipe = _make_pipeline(name, estimator)
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(model, X_test, y_test, problem_type):
    preds = model.predict(X_test)
    if problem_type == "classification":
        return float(f1_score(y_test, preds, average="macro", zero_division=0))
    return float(_safe_float(r2_score(y_test, preds)) or 0.0)


def run_automl(filename: str, target_column: str) -> dict:
    """End-to-end AutoML pipeline v5.6 (legacy entry point, fully preserved)."""
    file_path = Path("uploads") / filename
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    suffix = Path(filename).suffix.lower()
    _row_count_preview = None
    if suffix in (".csv", ".txt"):
        try:
            with open(file_path, "rb") as _f:
                _row_count_preview = sum(1 for _ in _f) - 1
        except Exception: pass
    if _row_count_preview is not None and _row_count_preview > _TIER3_MAX:
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=200_000, encoding_errors="replace", low_memory=False)
            df = pd.concat(chunk_iter, ignore_index=True)
        except Exception as exc:
            return {"error": f"Chunked file reading error: {exc}"}
    else:
        result = _load_file(file_path)
        if isinstance(result, dict): return result
        df = result
    if df.empty: return {"error": "Loaded file is empty."}

    n_total = len(df); tier = _get_tier(n_total)
    log.info(f"[Scale] {n_total:,} rows → Tier {tier} ({_TIER_NAMES[tier]})")
    small_dataset = n_total < 200
    data_warning  = ""
    if small_dataset:
        data_warning = (f"WARNING: Only {n_total} rows. "
                        f"XGBoost/LightGBM/CatBoost disabled; max_depth ≤ 5.")
        log.warning(f"[SmallDataset] {data_warning}")
    n_cv_folds = 10 if small_dataset else 3

    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^A-Za-z0-9_]", "", regex=True))
    target_column = re.sub(r"[^A-Za-z0-9_]", "", target_column.strip().replace(" ", "_"))
    if target_column not in df.columns:
        return {"error": f"Target '{target_column}' not found. Available: {df.columns.tolist()}"}

    try:
        df = universal_cleaning(df, target_column)
    except Exception as exc:
        return {"error": f"Cleaning failed: {exc}"}
    if df.shape[0] < 10: return {"error": "Fewer than 10 rows after cleaning."}

    if tier == 4: df = _frequency_encode(df, target_column)
    X = df.drop(columns=[target_column]); y = df[target_column]
    problem_type, detection_reason, y = _detect_problem_type(y, X.shape[1])
    log.info(f"[ProblemType] → {problem_type.upper()} ({detection_reason})")
    dataset_diagnostics = _log_dataset_diagnostics(df, target_column, problem_type)

    le = None
    if problem_type == "classification":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    if tier == 1:   X = pd.get_dummies(X, drop_first=True)
    elif tier == 2: X = _cap_cardinality(X, target_column, 50); X = pd.get_dummies(X, drop_first=True); X = X.astype("float32")
    elif tier == 3: X = _cap_cardinality(X, target_column, 20); X = pd.get_dummies(X, drop_first=True, sparse=False); X = X.astype("float32")
    else:           X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X, removed_features, leakage_detected = detect_data_leakage(X, y, threshold=0.90)
    X_train, X_test, y_train, y_test = split_data(X, y, problem_type, test_size=0.20)
    n_train = len(X_train)
    model_results        = {}
    feature_correlations = {}
    baseline_accuracy    = None
    _shap_samples        = {1: None, 2: 2_000, 3: 1_000, 4: 500}
    shap_n               = _shap_samples[tier]

    # Extract missing_pct for the scorer
    missing_pct_val = float(dataset_diagnostics.get("overall_missing_pct", 0.0) or 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════════
    if problem_type == "classification":
        baseline_accuracy = float(y_train.value_counts(normalize=True).max())
        n_classes = int(y.nunique())
        scoring   = "f1_macro" if n_classes > 2 else "f1"
        X_train, X_test, fs_report = auto_feature_selection(
            X_train, X_test, y_train, problem_type=problem_type,
            n_classes=n_classes, tier=tier)
        if tier >= 3:
            cv_sample_n = 50_000 if tier == 3 else 100_000
            X_cv, y_cv  = _downsample_for_cv(X_train, y_train, cv_sample_n, problem_type)
        else:
            X_cv, y_cv = X_train, y_train
        cv_splitter = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        candidates  = _build_cls_candidates(tier, small_dataset=small_dataset)
        candidates  = _apply_meta_learning_priority(
            candidates, n_rows=n_total, n_features=X_train.shape[1],
            problem_type=problem_type, missing_pct=missing_pct_val)
        rl_state = _build_rl_state(X_train, y_train, problem_type=problem_type,
                                   tier=tier, n_candidates=len(candidates))

        if tier == 4:
            X_cv_tr, X_cv_te, y_cv_tr, y_cv_te = train_test_split(
                X_cv, y_cv, test_size=0.2, random_state=42, stratify=y_cv)
            cv_scores, cv_std_scores, best_params_all = {}, {}, {}
            for name, (base_est, _) in candidates.items():
                try:
                    pipe  = _make_pipeline(name, base_est); pipe.fit(X_cv_tr, y_cv_tr)
                    score = float(f1_score(y_cv_te, pipe.predict(X_cv_te), average="macro", zero_division=0))
                    cv_scores[name] = score; cv_std_scores[name] = 0.0; best_params_all[name] = {}
                except Exception as exc:
                    log.warning(f"[CV-Class] {name} failed: {exc}")
            if not cv_scores: return {"error": "All classification CV evaluations failed."}
            best_name  = max(cv_scores, key=lambda n: cv_scores[n])
            best_model = _build_final_model(best_name, candidates[best_name][0], {})
            best_model.fit(X_train, y_train)
            agent_history = []; bandit_stats = {}; stacking_name = "N/A"
        else:
            (best_name, best_model, cv_scores, cv_std_scores,
             best_params_all, agent_history, bandit_stats) = _run_rl_multi_seed(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                candidates=candidates, problem_type=problem_type, tier=tier,
                scoring=scoring, n_folds=n_cv_folds, rl_state=rl_state)
            if best_model is None: return {"error": "RL pipeline: all model evaluations failed."}
            stack_model, stack_base_names = _build_stacking_model(
                cv_scores, candidates, best_params_all, problem_type, top_n=3)
            stacking_name = f"Stacking({'+'.join(stack_base_names)})"
            if stack_model is not None:
                try:
                    stack_model.fit(X_train, y_train)
                    stack_cv    = cross_val_score(stack_model, X_cv, y_cv,
                                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                                  scoring=scoring, n_jobs=-1)
                    stack_score = float(stack_cv.mean()); stack_std = float(stack_cv.std())
                    if stack_score > cv_scores.get(best_name, 0):
                        best_model = stack_model; cv_scores[stacking_name] = stack_score
                        cv_std_scores[stacking_name] = stack_std; best_params_all[stacking_name] = {}
                        best_name  = stacking_name
                except Exception as exc:
                    log.warning(f"[Stacking] Failed: {exc}"); stacking_name = "Stacking (failed)"

        for name in list(cv_scores.keys()):
            if name == stacking_name: continue
            try:
                mdl = (best_model if name == best_name else
                       _build_final_model(name, candidates[name][0], best_params_all.get(name, {})))
                if name != best_name: mdl.fit(X_train, y_train)
                preds_m = mdl.predict(X_test)
                model_results[name] = {
                    "accuracy": float(accuracy_score(y_test, preds_m)),
                    "f1_macro": float(f1_score(y_test, preds_m, average="macro", zero_division=0)),
                    "cv_mean":  cv_scores.get(name), "cv_std": cv_std_scores.get(name),
                    "scaled":   name in _NEEDS_SCALE or name in _NEEDS_NONNEG_SCALE,
                    "best_params": {k.replace("model__",""):v for k,v in best_params_all.get(name,{}).items()},
                }
            except Exception as exc:
                log.warning(f"[Model] {name} test eval failed: {exc}")

        preds       = best_model.predict(X_test)
        test_acc    = float(accuracy_score(y_test, preds))
        cm_array    = confusion_matrix(y_test, preds).tolist()
        cm_labels   = [str(c) for c in le.classes_] if le is not None else None
        train_preds = best_model.predict(X_train)
        train_acc   = float(accuracy_score(y_train, train_preds))
        cv_mean_best = cv_scores.get(best_name, 0.0)
        cv_std_best  = cv_std_scores.get(best_name, 0.0)
        overfit      = detect_overfitting(train_acc, cv_mean_best)
        confidence   = calculate_confidence(cv_mean_best, cv_std_best, n_total, overfit)

        roc_auc_val = None
        if n_classes == 2:
            try:
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(X_test)[:, 1]
                elif hasattr(best_model, "decision_function"):
                    proba = best_model.decision_function(X_test)
                else: proba = None
                if proba is not None: roc_auc_val = _safe_float(roc_auc_score(y_test, proba))
            except Exception: pass

        perf_report = {
            "all_model_scores": model_results, "accuracy": test_acc, "train_accuracy": train_acc,
            "cv_score_mean": cv_mean_best, "cv_score_std": cv_std_best, "cv_scoring": scoring,
            "n_cv_folds": n_cv_folds, "confusion_matrix": convert_to_python(cm_array),
            "confusion_matrix_labels": cm_labels, "roc_auc": roc_auc_val, "n_classes": n_classes,
            "n_train": n_train, "n_test": len(X_test), "problem_type_reason": detection_reason,
            "scaling_applied": best_name in _NEEDS_SCALE or best_name in _NEEDS_NONNEG_SCALE,
            "baseline_accuracy": baseline_accuracy, "scale_tier": tier, "scale_tier_name": _TIER_NAMES[tier],
            "best_params": {k.replace("model__",""):v for k,v in best_params_all.get(best_name,{}).items()},
            "feature_selection": fs_report, "overfitting": overfit,
            "confidence_score": confidence["score"], "confidence_label": confidence["label"],
            "leakage_detected": leakage_detected, "removed_features": removed_features,
            "data_warning": data_warning, "agent_history": agent_history, "bandit_stats": bandit_stats,
            "stacking_model": stacking_name,
            "baseline_alert": _check_baseline(best_name, cv_scores, baseline_accuracy, problem_type),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # REGRESSION
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        n_features       = X_train.shape[1]
        low_feature_mode = n_features <= 3
        if low_feature_mode:
            X_train_r, X_test_r, low_feat_note = _engineer_low_features(X_train, X_test, y_train)
        else:
            X_train_r, X_test_r = X_train.copy(), X_test.copy(); low_feat_note = ""
        try:
            corr_df = X_train.copy(); corr_df["__target__"] = y_train.values
            cv_corr = corr_df.corr()["__target__"].drop("__target__")
            feature_correlations = {c: _safe_float(v) for c, v in cv_corr.items()}
        except Exception as exc:
            log.warning(f"[Regression] Correlation failed: {exc}")
        X_train_r, X_test_r, fs_report = auto_feature_selection(
            X_train_r, X_test_r, y_train, problem_type=problem_type, n_classes=1, tier=tier)
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        if tier >= 3:
            cv_sample_n    = 50_000 if tier == 3 else 100_000
            X_cv_r, y_cv_r = _downsample_for_cv(X_train_r, y_train, cv_sample_n, problem_type)
        else:
            X_cv_r, y_cv_r = X_train_r, y_train
        candidates = _build_reg_candidates(tier, low_feature_mode, small_dataset=small_dataset)
        candidates = _apply_meta_learning_priority(
            candidates, n_rows=n_total, n_features=X_train_r.shape[1],
            problem_type=problem_type, missing_pct=missing_pct_val)
        rl_state   = _build_rl_state(X_train_r, y_train, problem_type=problem_type,
                                     tier=tier, n_candidates=len(candidates))

        if tier == 4:
            X_cvr_tr, X_cvr_te, y_cvr_tr, y_cvr_te = train_test_split(
                X_cv_r, y_cv_r, test_size=0.2, random_state=42)
            cv_scores_r, cv_std_r, best_params_all = {}, {}, {}
            for name, (base_est, _) in candidates.items():
                try:
                    pipe = _make_pipeline(name, base_est); pipe.fit(X_cvr_tr, y_cvr_tr)
                    score = _safe_float(r2_score(y_cvr_te, pipe.predict(X_cvr_te)))
                    cv_scores_r[name] = score or 0.0; cv_std_r[name] = 0.0; best_params_all[name] = {}
                except Exception as exc:
                    log.warning(f"[CV-Reg] {name} failed: {exc}")
            if not cv_scores_r: return {"error": "All regression CV evaluations failed."}
            best_name  = max(cv_scores_r, key=lambda n: cv_scores_r[n])
            best_model = _build_final_model(best_name, candidates[best_name][0], {})
            best_model.fit(X_train_r, y_train)
            agent_history = []; bandit_stats = {}; stacking_name = "N/A"
        else:
            (best_name, best_model, cv_scores_r, cv_std_r,
             best_params_all, agent_history, bandit_stats) = _run_rl_multi_seed(
                X_train=X_train_r, X_test=X_test_r, y_train=y_train, y_test=y_test,
                candidates=candidates, problem_type=problem_type, tier=tier,
                scoring="r2", n_folds=n_cv_folds, rl_state=rl_state)
            if best_model is None: return {"error": "RL pipeline: all model evaluations failed."}
            stack_model, stack_base_names = _build_stacking_model(
                cv_scores_r, candidates, best_params_all, problem_type, top_n=3)
            stacking_name = f"Stacking({'+'.join(stack_base_names)})" if stack_base_names else "N/A"
            if stack_model is not None:
                try:
                    stack_model.fit(X_train_r, y_train)
                    stack_cv    = cross_val_score(stack_model, X_cv_r, y_cv_r,
                                                  cv=KFold(n_splits=3, shuffle=True, random_state=42),
                                                  scoring="r2", n_jobs=-1)
                    stack_score = float(stack_cv.mean()); stack_std = float(stack_cv.std())
                    if stack_score > cv_scores_r.get(best_name, 0):
                        best_model = stack_model; cv_scores_r[stacking_name] = stack_score
                        cv_std_r[stacking_name] = stack_std; best_params_all[stacking_name] = {}
                        best_name = stacking_name
                except Exception as exc:
                    log.warning(f"[Stacking] Failed: {exc}")

        for name in list(cv_scores_r.keys()):
            if name == stacking_name: continue
            try:
                mdl = (best_model if name == best_name else
                       _build_final_model(name, candidates[name][0], best_params_all.get(name, {})))
                if name != best_name: mdl.fit(X_train_r, y_train)
                p    = mdl.predict(X_test_r)
                model_results[name] = {
                    "R2": _safe_float(r2_score(y_test, p)),
                    "MAE": _safe_float(mean_absolute_error(y_test, p)),
                    "RMSE": _safe_float(float(np.sqrt(mean_squared_error(y_test, p)))),
                    "cv_r2_mean": cv_scores_r.get(name), "cv_r2_std": cv_std_r.get(name),
                    "scaled": name in _NEEDS_SCALE,
                    "best_params": {k.replace("model__",""):v for k,v in best_params_all.get(name,{}).items()},
                }
            except Exception as exc:
                log.warning(f"[Model] {name} test eval failed: {exc}")

        preds     = best_model.predict(X_test_r)
        test_r2   = _safe_float(r2_score(y_test, preds))
        test_mae  = _safe_float(mean_absolute_error(y_test, preds))
        test_rmse = _safe_float(float(np.sqrt(mean_squared_error(y_test, preds))))
        tr_preds  = best_model.predict(X_train_r)
        train_r2  = _safe_float(r2_score(y_train, tr_preds))
        cv_r2_best  = cv_scores_r.get(best_name, 0.0)
        cv_std_best = cv_std_r.get(best_name, 0.0)
        overfit     = detect_overfitting(train_r2 or 0.0, cv_r2_best)
        confidence  = calculate_confidence(cv_r2_best, cv_std_best, n_total, overfit)

        perf_report = {
            "all_model_scores": model_results, "R2": test_r2, "MAE": test_mae, "RMSE": test_rmse,
            "train_R2": train_r2, "cv_r2_mean": cv_r2_best, "cv_r2_std": cv_std_best,
            "cv_score_mean": cv_r2_best, "cv_score_std": cv_std_best, "n_cv_folds": n_cv_folds,
            "feature_correlations": feature_correlations, "n_train": n_train, "n_test": len(X_test),
            "problem_type_reason": detection_reason, "scaling_applied": best_name in _NEEDS_SCALE,
            "scale_tier": tier, "scale_tier_name": _TIER_NAMES[tier],
            "best_params": {k.replace("model__",""):v for k,v in best_params_all.get(best_name,{}).items()},
            "feature_selection": fs_report, "overfitting": overfit,
            "confidence_score": confidence["score"], "confidence_label": confidence["label"],
            "leakage_detected": leakage_detected, "removed_features": removed_features,
            "data_warning": data_warning, "agent_history": agent_history, "bandit_stats": bandit_stats,
            "stacking_model": stacking_name,
            "baseline_alert": _check_baseline(best_name, cv_scores_r, 0.0, problem_type),
            **({
                "low_feature_mode": True, "low_feature_note": low_feat_note,
                "engineered_features": X_train_r.columns.tolist(),
            } if low_feature_mode else {}),
        }
        X_train, X_test = X_train_r, X_test_r

    quality     = _quality_commentary(problem_type=problem_type,
                                      accuracy=perf_report.get("accuracy"),
                                      r2=perf_report.get("R2"), mae=perf_report.get("MAE"),
                                      baseline_acc=baseline_accuracy,
                                      n_classes=perf_report.get("n_classes", 2))
    feat_imp    = _get_feature_importance(best_model, X_test.columns.tolist(),
                                          fs_report=perf_report.get("feature_selection"))
    chart_fname = _make_feature_chart(feat_imp, feature_correlations, best_name, problem_type)
    sample_explanation = _get_shap_explanation(best_model, X_train, X_test, shap_sample_n=shap_n)
    model_fname = _save_model(best_model, X_test.columns.tolist(), problem_type, label_encoder=le)
    pdf_fname   = _save_pdf(best_name, problem_type, model_results, perf_report, chart_fname, quality)

    if problem_type == "classification":
        score_str = (f"CV {perf_report['cv_scoring']}: {perf_report['cv_score_mean']:.4f} "
                     f"±{perf_report['cv_score_std']:.4f}  |  Test Acc: {perf_report['accuracy']:.4f}")
    else:
        r2_val = perf_report.get("R2")
        score_str = (f"CV R²: {perf_report['cv_r2_mean']:.4f} ±{perf_report['cv_r2_std']:.4f}  |  "
                     f"Test R²: {r2_val:.4f}" if r2_val is not None else "CV R²: N/A")

    return convert_to_python({
        "problem_type":       problem_type,
        "best_model":         best_name,
        "performance":        perf_report,
        "model_quality":      quality,
        "model_name":         model_fname,
        "pdf_report":         pdf_fname,
        "feature_chart":      chart_fname,
        "scale_tier":         tier,
        "scale_tier_name":    _TIER_NAMES[tier],
        "feature_selection":  convert_to_python(perf_report["feature_selection"]),
        "sample_explanation": sample_explanation,
        "friendly_summary":   (f"[Tier {tier} – {_TIER_NAMES[tier]}: {n_total:,} rows]  "
                               f"Best model: {best_name}. {score_str}. "
                               f"Quality: {quality['rating']} — {quality['summary']}"),
        "cv_score_mean":      perf_report.get("cv_score_mean"),
        "cv_score_std":       perf_report.get("cv_score_std"),
        "test_score":         perf_report.get("accuracy") or perf_report.get("R2"),
        "overfitting":        perf_report.get("overfitting", False),
        "confidence_score":   perf_report.get("confidence_score"),
        "confidence_label":   perf_report.get("confidence_label"),
        "leakage_detected":   leakage_detected,
        "removed_features":   removed_features,
        "agent_history":      agent_history,
        "bandit_stats":       bandit_stats,
        "data_warning":       data_warning,
        "stacking_model":     stacking_name,
        "stability_seeds":    _STABILITY_SEEDS if tier in _STABILITY_TIERS else [],
        "baseline_alert":     perf_report.get("baseline_alert", {}),
        "dataset_diagnostics":dataset_diagnostics,
        "decision":           None, "meta_insight": None, "retrain_decision": None,
        "pipeline_stage_logs":[], "model_metrics": None, "prediction": None,
    })


# ==============================================================================
# PATCH-B — run_automl_extended  (v5.6 recommended entry point)
# ==============================================================================

_patch_log = logging.getLogger(__name__)


def run_automl_extended(filename, target_column, use_new_pipeline=True,
                        time_budget=None, context_overrides=None,
                        save_after_run=True) -> dict:
    t0 = time.perf_counter()
    if use_new_pipeline:
        _avail = globals().get("_AGENT_INTEGRATION_AVAILABLE", False)
        if _avail:
            try:
                result = run_automl_with_agents(
                    filename=filename, target_column=target_column,
                    time_budget=time_budget, context_overrides=context_overrides,
                    save_after_run=save_after_run)
                if "error" not in result:
                    result["_pipeline_path"] = "modular_v5.6"
                    result["_total_wall_s"]  = round(time.perf_counter() - t0, 3)
                    return result
            except Exception as exc:
                _patch_log.warning(f"[Extended] Modular pipeline error ({exc}), falling back.")
    result = run_automl(filename, target_column)
    if "error" in result: return result
    result.setdefault("decision",            None)
    result.setdefault("meta_insight",        None)
    result.setdefault("retrain_decision",    {"should_retrain": False, "reason": "legacy_path"})
    result.setdefault("pipeline_stage_logs", [])
    result.setdefault("model_metrics",       result.get("performance"))
    result.setdefault("prediction", {
        "problem_type": result.get("problem_type"),
        "best_model":   result.get("best_model"),
        "score":        (result.get("performance", {}).get("accuracy") or
                         result.get("performance", {}).get("R2")),
        "confidence":   result.get("confidence_label"),
    })
    result["_pipeline_path"] = "legacy_v5.2"
    result["_total_wall_s"]  = round(time.perf_counter() - t0, 3)
    return result


# ==============================================================================
# SECTION 13 — AGENT SYSTEM INTEGRATION
# ==============================================================================

try:
    from automl_integration import (
        run_automl_with_agents, record_outcome, agent_status,
        save_agents, reset_agent_system,
    )
    _AGENT_INTEGRATION_AVAILABLE = True
    log.info("[v5.6] Agent integration layer loaded.")
except ImportError as _agent_import_err:
    _AGENT_INTEGRATION_AVAILABLE = False
    log.info(f"[v5.6] Agent integration not available ({_agent_import_err}).")

    def run_automl_with_agents(filename, target_column, time_budget=None,
                                context_overrides=None, save_after_run=True) -> dict:
        return run_automl_extended(filename=filename, target_column=target_column,
                                   use_new_pipeline=False, time_budget=time_budget,
                                   context_overrides=context_overrides,
                                   save_after_run=save_after_run)

    def record_outcome(*args, **kwargs) -> dict:
        return {"status": "agents_unavailable"}

    def agent_status() -> dict:
        return {"agents_available": False}

    def save_agents() -> dict:
        return {"status": "agents_unavailable"}

    def reset_agent_system() -> dict:
        return {"status": "agents_unavailable"}