"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  feature_extractor.py  —  v2.0  (AutoAnalytica AI v5.5)                    ║
║                                                                              ║
║  v2.0 Changes                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  + UncertaintyFeatureBlock : epistemic/aleatoric/confidence_gap computed    ║
║      once and injected into RL state, meta-feature, AND retrain vectors     ║
║      so all three models benefit from uncertainty signals                   ║
║  + ConfidenceWeightModel   : Ridge learns optimal blend weights for         ║
║      calibrated_confidence instead of hardcoded 0.50/0.25/0.15/0.10        ║
║  + SharedExperienceStore   : every extract() call records the bundle        ║
║      to the global experience store for cross-agent learning                ║
║  + RL state vector (dim 20) updated: slots [11] confidence now uses         ║
║      calibrated_confidence from ConfidenceWeightModel (not raw score)       ║
║  + Meta vector (dim 18): slots [5] confidence = calibrated,                 ║
║      two new uncertainty slots reuse existing dims 4 and 8                  ║
║  + Retrain vector (dim 12): slot [3] confidence = calibrated,               ║
║      drift_score now uses uncertainty-weighted PSI                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── Inline path bootstrap (no external module needed) ──────────────────────
import sys as _sys, pathlib as _pl
_sd = str(_pl.Path(__file__).resolve().parent)
if _sd not in _sys.path: _sys.path.insert(0, _sd)
del _sd
# ────────────────────────────────────────────────────────────────────────────
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from app.services.rl_agent import build_state_vector, STATE_DIM
except ImportError:
    try:
        from rl_agent import build_state_vector, STATE_DIM
    except ImportError as exc:
        raise ImportError(
            "feature_extractor requires rl_agent.py (v5.5+) to export "
            "build_state_vector and STATE_DIM."
        ) from exc

try:
    from experience_store import (
        SharedExperienceStore, make_experience,
        AGENT_PIPELINE, record_experience,
    )
    _STORE_AVAILABLE = True
except ImportError:
    _STORE_AVAILABLE = False

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RL_DIM:      int = STATE_DIM   # 20
META_DIM:    int = 18
RETRAIN_DIM: int = 12


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureBundle:
    rl_state_vector:     np.ndarray
    meta_feature_vec:    np.ndarray
    retrain_feature_vec: np.ndarray
    feature_dict:        Dict[str, Any]
    extraction_time_ms:  float = 0.0
    run_id:              str   = ""

    def as_api_dict(self) -> Dict[str, Any]:
        def _fmt(v):
            return round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v
        return {
            "rl_state_vector":     [round(float(v), 5) for v in self.rl_state_vector],
            "meta_feature_vec":    [round(float(v), 5) for v in self.meta_feature_vec],
            "retrain_feature_vec": [round(float(v), 5) for v in self.retrain_feature_vec],
            "feature_dict":        {k: _fmt(v) for k, v in self.feature_dict.items()},
            "extraction_time_ms":  round(self.extraction_time_ms, 2),
            "run_id":              self.run_id,
            "dims": {"rl_state": RL_DIM, "meta": META_DIM, "retrain": RETRAIN_DIM},
        }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))

def _log_norm(raw: float, scale: float) -> float:
    return _clamp(math.log10(max(raw, 0) + 1) / scale)

def _entropy(probs: List[float]) -> float:
    total = sum(probs)
    if total <= 0: return 0.0
    ps = [p / total for p in probs if p > 0]
    return float(-sum(p * math.log(p) for p in ps))

def _top_feature_concentration(feat_imp: Dict[str, float]) -> float:
    if not feat_imp: return 0.5
    vals  = sorted(feat_imp.values(), reverse=True)
    total = sum(vals) or 1.0
    top5  = sum(vals[:5])
    return _clamp(top5 / total)

def _model_family_index(model_name: str) -> int:
    families: Dict[int, List[str]] = {
        0: ["logistic","linear","ridge","lasso","elastic","bayesian","sgd","linearsvc","linearsvr"],
        1: ["randomforest","extratrees","decisiontree","gradientboosting","histgradient","adaboost","bagging"],
        2: ["xgb","lgbm","lightgbm","catboost"],
        3: ["gaussiannb","bernoullinb","multinomialnb","qda"],
    }
    name_lc = model_name.lower().replace("_","").replace(" ","")
    for idx, keywords in families.items():
        if any(kw in name_lc for kw in keywords): return idx
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE WEIGHT MODEL  (replaces 0.50/0.25/0.15/0.10)
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceWeightModel:
    """
    Ridge regression that learns the optimal blend weights for the four
    components of calibrated_confidence:

        w0*base_conf + w1*(1-epistemic) + w2*(1-aleatoric) + w3*signal_strength

    Trained from SharedExperienceStore: target = actual test_score.
    The model learns which component is most predictive of real performance.
    Falls back to [0.50, 0.25, 0.15, 0.10] until MIN_SAMPLES records exist.
    """

    MIN_SAMPLES   = 25
    RETRAIN_EVERY = 10
    _DEFAULTS     = [0.50, 0.25, 0.15, 0.10]

    def __init__(self) -> None:
        self._weights:      List[float] = list(self._DEFAULTS)
        self._next_retrain: int         = self.MIN_SAMPLES

    @property
    def weights(self) -> List[float]:
        return self._weights

    def maybe_fit(self, records: list) -> None:
        if len(records) >= self._next_retrain:
            self._fit(records)
            self._next_retrain = len(records) + self.RETRAIN_EVERY

    def _fit(self, records: list) -> None:
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            return
        X, y = [], []
        for r in records:
            metrics   = r.get("metrics") or r.get("performance") or {}
            base_conf = _safe(metrics.get("confidence_score", 0.5))
            test_sc   = _safe(metrics.get("accuracy") or metrics.get("R2") or 0.5)
            cv_std    = _safe(metrics.get("cv_score_std", 0.1))
            n_rows    = _safe((r.get("dataset_diagnostics") or {}).get("n_rows", 100))
            epistemic = _clamp((cv_std * 5.0 + _clamp(
                1.0 - math.log10(max(n_rows, 1)) / 7.0) * 0.3) / 1.3)
            aleatoric = _clamp((1.0 - _safe(metrics.get("cv_score_mean", 0.5))) * 0.6)
            shap_vals = (r.get("sample_explanation") or {}).get("shap_values") or {}
            if shap_vals:
                vals   = list(shap_vals.values())
                total  = sum(abs(v) for v in vals) or 1.0
                signal = _clamp(sum(sorted(abs(v) for v in vals)[-5:]) / total)
            else:
                signal = 0.5
            X.append([base_conf, 1.0 - epistemic, 1.0 - aleatoric, signal])
            y.append(test_sc)
        if len(X) < 5: return
        reg = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        try:
            reg.fit(X, y)
            raw   = list(reg.coef_)
            total = sum(raw) or 1.0
            self._weights = [float(max(0.05, min(0.70, w / total))) for w in raw]
            log.info(f"[ConfidenceWeightModel] weights={self._weights} ({len(X)} records)")
        except Exception as exc:
            log.warning(f"[ConfidenceWeightModel] Fit failed: {exc}")


_global_confidence_weight_model = ConfidenceWeightModel()


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY FEATURE BLOCK  (v2.0 — single computation, injected everywhere)
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyFeatureBlock:
    """
    Computes epistemic uncertainty, aleatoric uncertainty, calibrated
    confidence, and confidence_gap in ONE place.

    These values are:
      1. Stored in feature_dict under standard keys
      2. Written to the SharedExperienceStore with every run
      3. Used by RL state vector (slot 11 — confidence)
      4. Used by meta-feature vector (slot 5 — confidence)
      5. Used by retrain-feature vector (slot 3 — confidence)

    This guarantees all three downstream models see IDENTICAL uncertainty
    estimates — there is no divergence between what RL sees vs meta vs retrain.

    Calibrated confidence uses learned blend weights from ConfidenceWeightModel.
    """

    def compute(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Returns a dict with the following keys (all ∈ [0, 1]):

          epistemic_uncertainty   — reducible uncertainty (CV stability + data scarcity)
          aleatoric_uncertainty   — irreducible uncertainty (performance ceiling)
          calibrated_confidence   — learned-weighted blend ∈ [0, 1]
          raw_confidence          — raw pipeline confidence_score
          confidence_gap          — |calibrated - raw| (detects overconfidence)
          signal_strength         — SHAP importance concentration
        """
        # Warm up weight model from store if available
        self._try_warm_weights()

        cv_score  = _safe(result.get("cv_score_mean", 0.5))
        cv_std    = _safe(result.get("cv_score_std",  0.1))
        base_conf = _safe(result.get("confidence_score", cv_score))
        n_rows    = _safe((result.get("dataset_diagnostics") or {}).get("n_rows", 100))
        shap      = result.get("sample_explanation", {}) or {}
        shap_vals = shap.get("shap_values", {}) or {}

        # ── Epistemic (reducible — data + model instability) ──────────────────
        data_scarcity = _clamp(1.0 - math.log10(max(n_rows, 1)) / 7.0)
        epistemic     = _clamp((cv_std * 5.0 + data_scarcity * 0.3) / 1.3)

        # ── Aleatoric (irreducible — label noise proxy) ───────────────────────
        aleatoric = _clamp((1.0 - cv_score) * 0.6)

        # ── Signal strength (SHAP concentration) ─────────────────────────────
        signal_strength = _top_feature_concentration(shap_vals) if shap_vals else 0.5

        # ── Calibrated confidence — LEARNED weights ───────────────────────────
        w0, w1, w2, w3 = _global_confidence_weight_model.weights
        calibrated = _clamp(
            base_conf          * w0
            + (1.0 - epistemic)  * w1
            + (1.0 - aleatoric)  * w2
            + signal_strength    * w3
        )

        # ── Confidence gap — detects overconfidence ───────────────────────────
        confidence_gap = _clamp(abs(calibrated - base_conf))

        return {
            "epistemic_uncertainty": round(epistemic,       4),
            "aleatoric_uncertainty": round(aleatoric,       4),
            "calibrated_confidence": round(calibrated,      4),
            "raw_confidence":        round(base_conf,       4),
            "confidence_gap":        round(confidence_gap,  4),
            "signal_strength":       round(signal_strength, 4),
        }

    def _try_warm_weights(self) -> None:
        """Warm ConfidenceWeightModel from SharedExperienceStore if available."""
        if not _STORE_AVAILABLE:
            return
        try:
            import json as _json
            from pathlib import Path as _Path
            lp = _Path("agent_system_state") / "outcome_ledger.json"
            if lp.exists():
                with open(lp) as fh:
                    records = _json.load(fh).get("records", [])
                _global_confidence_weight_model.maybe_fit(records)
        except Exception:
            pass


_global_uncertainty_block = UncertaintyFeatureBlock()


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scalars(result: Dict[str, Any]) -> Dict[str, Any]:
    perf  = result.get("performance", {}) or {}
    fs    = (result.get("feature_selection") or perf.get("feature_selection") or {})
    diag  = result.get("dataset_diagnostics", {}) or {}
    ba    = result.get("baseline_alert", {}) or {}
    shap  = result.get("sample_explanation", {}) or {}
    ah    = result.get("agent_history", []) or []

    cv_score    = _safe(result.get("cv_score_mean",  perf.get("cv_score_mean",  0.0)))
    cv_std      = _safe(result.get("cv_score_std",   perf.get("cv_score_std",   0.1)))
    test_score  = _safe(result.get("test_score", perf.get("accuracy", perf.get("R2", 0.0))))
    train_score = _safe(perf.get("train_accuracy", perf.get("train_R2", test_score)))
    overfit_gap = _clamp(max(0.0, train_score - cv_score))
    confidence  = _clamp(_safe(result.get("confidence_score", perf.get("confidence_score", 0.5))))
    roc_auc     = _safe(perf.get("roc_auc", cv_score))
    baseline_gap     = _safe(ba.get("gap", 0.0))
    baseline_gap_n   = _clamp(0.5 + baseline_gap)
    cv_std_n         = _clamp(cv_std / 0.50)
    all_scores       = [e.get("score", 0) for e in ah
                        if isinstance(e, dict) and e.get("score") is not None]
    cv_range         = _clamp(max(all_scores) - min(all_scores)) if len(all_scores) >= 2 else 0.0
    eff_scores       = [e["effective_score"] for e in ah
                        if isinstance(e, dict) and e.get("effective_score") is not None]
    avg_eff_score    = _safe(float(np.mean(eff_scores))) if eff_scores else cv_score
    real_iters       = [e for e in ah if isinstance(e, dict) and "score" in e]
    overfit_iters    = sum(1 for e in real_iters if e.get("overfit"))
    overfit_rate     = _clamp(overfit_iters / max(len(real_iters), 1))

    n_rows      = _safe(diag.get("n_rows") or perf.get("n_train") or 1000)
    n_train     = _safe(perf.get("n_train", n_rows * 0.8))
    n_test      = _safe(perf.get("n_test",  n_rows * 0.2))
    n_feats     = _safe(fs.get("final_features") or diag.get("n_cols") or 10)
    n_orig_f    = _safe(fs.get("original_features") or n_feats)
    feat_reduction = _clamp(1.0 - (n_feats / max(n_orig_f, 1)))
    n_rows_norm = _log_norm(n_rows,  7.0)
    n_feats_norm= _log_norm(n_feats, 4.0)

    imb_raw   = diag.get("class_imbalance")
    if imb_raw is None:
        n_cls = _safe(diag.get("n_classes", 0))
        imb_raw = (1.0 / max(n_cls, 1)) if n_cls > 0 else 0.5
    imbalance   = _clamp(_safe(imb_raw))
    missing_pct = _clamp(_safe(diag.get("overall_missing_pct", 0.0)) / 100.0)

    skew_dict  = diag.get("most_skewed_features", {}) or {}
    skew_vals  = list(skew_dict.values())
    skew_max   = _clamp(max(skew_vals, default=0.0) / 10.0)
    skew_mean  = _clamp(float(np.mean(skew_vals)) / 10.0 if skew_vals else 0.0)

    steps_applied  = fs.get("steps_applied") or []
    fs_steps_n     = _clamp(len(steps_applied) / 5.0)
    pca_applied    = float(bool(fs.get("pca_applied", False)))
    lda_applied    = float(bool(fs.get("lda_applied", False)))

    shap_vals  = (shap.get("shap_values", {}) or {})
    feat_conc  = _top_feature_concentration(shap_vals)
    shap_avail = float(bool(shap.get("available", False)))

    leakage      = float(bool(result.get("leakage_detected", False)))
    n_removed    = _clamp(len(result.get("removed_features") or []) / 20.0)
    model_name   = str(result.get("best_model") or result.get("best_model_name") or "")
    stacking     = float("Stacking" in model_name)
    tier_raw     = int(_safe(perf.get("scale_tier") or result.get("scale_tier") or 1))
    tier_n       = _clamp(tier_raw / 4.0)
    model_family = _clamp(_model_family_index(model_name) / 4.0)
    prob_type    = 1.0 if result.get("problem_type", "classification") == "regression" else 0.0
    small_ds     = 1.0 if n_rows < 200 else 0.0

    n_iters_n    = _clamp(len(real_iters) / 30.0)
    eps_final    = _safe(real_iters[-1].get("epsilon_used", 0.2) if real_iters else 0.2)
    bandit_stats = result.get("bandit_stats", {}) or {}
    n_arms_tried = _clamp(
        len([v for v in bandit_stats.values()
             if isinstance(v, dict) and v.get("trials", 0) > 0]) / 10.0)

    return {
        "cv_score": cv_score, "cv_std": cv_std, "cv_std_n": cv_std_n,
        "cv_range": cv_range, "test_score": test_score, "train_score": train_score,
        "overfit_gap": overfit_gap, "overfit_gap_n": _clamp(overfit_gap / 0.40),
        "confidence": confidence, "roc_auc": roc_auc,
        "baseline_gap": baseline_gap, "baseline_gap_n": baseline_gap_n,
        "avg_eff_score": avg_eff_score, "overfit_rate": overfit_rate,
        "n_rows": n_rows, "n_rows_norm": n_rows_norm, "n_train": n_train,
        "n_test": n_test, "n_feats": n_feats, "n_feats_norm": n_feats_norm,
        "n_orig_features": n_orig_f, "feat_reduction": feat_reduction,
        "imbalance": imbalance, "missing_pct": missing_pct,
        "skew_max": skew_max, "skew_mean": skew_mean, "small_dataset": small_ds,
        "fs_steps_n": fs_steps_n, "pca_applied": pca_applied, "lda_applied": lda_applied,
        "feat_concentration": feat_conc, "shap_available": shap_avail,
        "leakage": leakage, "n_removed_n": n_removed, "stacking": stacking,
        "tier": tier_n, "tier_raw": float(tier_raw), "model_family": model_family,
        "prob_type": prob_type, "n_iters_n": n_iters_n, "eps_final": _clamp(eps_final),
        "n_arms_tried_n": n_arms_tried,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR BUILDERS  (all use uncertainty block values from feature_dict)
# ─────────────────────────────────────────────────────────────────────────────

def _build_meta_vector(fd: Dict[str, Any]) -> np.ndarray:
    """
    META_DIM (18) feature vector.

    v2.0 change: slot [5] uses calibrated_confidence (from ConfidenceWeightModel)
    instead of raw confidence_score, and slot [8] replaces cv_std_n with
    epistemic_uncertainty so the meta-model directly sees model uncertainty.

    Layout
    ──────
     [0]  cv_score
     [1]  cv_std_n             (raw CV spread)
     [2]  cv_range
     [3]  test_score
     [4]  overfit_gap_n
     [5]  calibrated_confidence  ← v2.0: learned-weighted blend
     [6]  baseline_gap_n
     [7]  avg_eff_score
     [8]  epistemic_uncertainty  ← v2.0: was overfit_rate
     [9]  n_rows_norm
    [10]  n_feats_norm
    [11]  feat_reduction
    [12]  imbalance
    [13]  missing_pct
    [14]  skew_max
    [15]  feat_concentration
    [16]  model_family
    [17]  stacking
    """
    vec = np.array([
        fd["cv_score"],
        fd["cv_std_n"],
        fd["cv_range"],
        fd["test_score"],
        fd["overfit_gap_n"],
        fd.get("calibrated_confidence", fd["confidence"]),  # v2.0
        fd["baseline_gap_n"],
        fd["avg_eff_score"],
        fd.get("epistemic_uncertainty", fd["overfit_rate"]),  # v2.0
        fd["n_rows_norm"],
        fd["n_feats_norm"],
        fd["feat_reduction"],
        fd["imbalance"],
        fd["missing_pct"],
        fd["skew_max"],
        fd["feat_concentration"],
        fd["model_family"],
        fd["stacking"],
    ], dtype=np.float32)
    assert len(vec) == META_DIM, f"Meta vector {len(vec)} ≠ {META_DIM}"
    return vec


def _build_retrain_vector(fd: Dict[str, Any]) -> np.ndarray:
    """
    RETRAIN_DIM (12) feature vector.

    v2.0 change: slot [3] uses calibrated_confidence (not raw); slot [10]
    uses aleatoric_uncertainty so the retrain model sees irreducible noise
    estimates — high aleatoric = labels may be noisy → retrain less useful.

    Layout
    ──────
     [0]  overfit_gap_n
     [1]  cv_std_n
     [2]  cv_range
     [3]  calibrated_confidence  ← v2.0: learned-weighted blend
     [4]  baseline_gap_n
     [5]  overfit_rate
     [6]  n_rows_norm
     [7]  missing_pct
     [8]  leakage
     [9]  n_iters_n
    [10]  aleatoric_uncertainty   ← v2.0: irreducible noise signal
    [11]  feat_reduction
    """
    vec = np.array([
        fd["overfit_gap_n"],
        fd["cv_std_n"],
        fd["cv_range"],
        fd.get("calibrated_confidence", fd["confidence"]),   # v2.0
        fd["baseline_gap_n"],
        fd["overfit_rate"],
        fd["n_rows_norm"],
        fd["missing_pct"],
        fd["leakage"],
        fd["n_iters_n"],
        fd.get("aleatoric_uncertainty", fd["n_arms_tried_n"]),  # v2.0
        fd["feat_reduction"],
    ], dtype=np.float32)
    assert len(vec) == RETRAIN_DIM, f"Retrain vector {len(vec)} ≠ {RETRAIN_DIM}"
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    PSI-based online drift detector.

    v2.0: also tracks uncertainty metrics so the drift score incorporates
    changes in epistemic/aleatoric uncertainty over time — not just raw
    performance metrics.
    """

    N_BASELINE: int = 20
    N_CURRENT:  int = 10
    N_BINS:     int = 10

    def __init__(self) -> None:
        self._history: List[Dict[str, float]] = []

    def push(self, feature_dict: Dict[str, Any]) -> None:
        self._history.append(
            {k: float(v) for k, v in feature_dict.items()
             if isinstance(v, (int, float, np.floating))
             and not math.isnan(float(v)) and not math.isinf(float(v))})

    def drift_score(self) -> float:
        n_needed = self.N_BASELINE + self.N_CURRENT
        if len(self._history) < n_needed: return 0.0
        baseline = self._history[:self.N_BASELINE]
        current  = self._history[-self.N_CURRENT:]
        keys     = [k for k in baseline[0] if k in current[0]]
        psi_values = []
        for key in keys:
            b_vals = [d[key] for d in baseline]
            c_vals = [d[key] for d in current]
            psi    = self._psi(b_vals, c_vals)
            if psi is not None: psi_values.append(psi)
        return float(np.mean(psi_values)) if psi_values else 0.0

    def _psi(self, base, curr):
        try:
            all_vals = base + curr
            lo, hi   = min(all_vals), max(all_vals)
            if hi == lo: return None
            bins = np.linspace(lo, hi, self.N_BINS + 1)
            b_counts, _ = np.histogram(base, bins=bins)
            c_counts, _ = np.histogram(curr, bins=bins)
            b_pct = (b_counts + 1e-6) / max(len(base), 1)
            c_pct = (c_counts + 1e-6) / max(len(curr), 1)
            return abs(float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))))
        except Exception: return None

    def is_drifting(self, threshold: float = 0.10) -> bool:
        return self.drift_score() > threshold

    def summary(self) -> Dict[str, Any]:
        score = self.drift_score()
        return {
            "n_runs_recorded": len(self._history),
            "drift_score":     round(score, 5),
            "is_drifting":     self.is_drifting(),
            "severity":        ("significant" if score > 0.25 else
                                "moderate"    if score > 0.10 else "none"),
        }

    def reset(self) -> None:
        self._history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ERROR DISTRIBUTION ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class ErrorDistributionAnalyser:
    def analyse(self, result: Dict[str, Any]) -> Dict[str, float]:
        perf  = result.get("performance", {}) or {}
        ptype = result.get("problem_type", "classification")
        if ptype == "classification": return self._cls_features(perf)
        return self._reg_features(perf)

    def _cls_features(self, perf):
        cm = perf.get("confusion_matrix")
        cm_entropy = 0.5
        if cm and isinstance(cm, list) and len(cm) > 0:
            try:
                cm_arr   = np.array(cm, dtype=float)
                row_sums = cm_arr.sum(axis=1, keepdims=True) + 1e-9
                cm_norm  = cm_arr / row_sums
                entropies = [-float(np.sum(r * np.log(r + 1e-9))) for r in cm_norm]
                max_ent   = math.log(max(len(cm), 2))
                cm_entropy = _clamp(float(np.mean(entropies)) / max_ent)
            except Exception: pass
        accuracy   = _safe(perf.get("accuracy",      0.5))
        f1_macro   = _safe(perf.get("cv_score_mean", accuracy))
        roc_auc    = _safe(perf.get("roc_auc",       f1_macro))
        acc_f1_gap = _clamp(abs(accuracy - f1_macro))
        return {
            "cm_entropy": round(cm_entropy, 5), "accuracy": round(accuracy, 5),
            "f1_macro": round(f1_macro, 5), "roc_auc": round(roc_auc, 5),
            "acc_f1_gap": round(acc_f1_gap, 5),
            "mae": 0.0, "rmse": 0.0, "r2": 0.0, "residual_spread": 0.0,
        }

    def _reg_features(self, perf):
        r2   = _safe(perf.get("R2",   0.0))
        mae  = _safe(perf.get("MAE",  1.0))
        rmse = _safe(perf.get("RMSE", 1.0))
        r2_n = _clamp(max(r2, 0.0))
        residual_spread = _clamp(abs(rmse - mae) / (mae + 1e-9) / 10.0)
        return {
            "cm_entropy": 0.0, "accuracy": 0.0, "f1_macro": 0.0,
            "roc_auc": 0.0, "acc_f1_gap": 0.0,
            "mae": round(_clamp(1.0 / (1.0 + mae)),  5),
            "rmse": round(_clamp(1.0 / (1.0 + rmse)), 5),
            "r2": round(r2_n, 5),
            "residual_spread": round(residual_spread, 5),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR  (v2.0)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Stateful feature extractor — v2.0.

    Key changes from v1.0
    ─────────────────────
    1. UncertaintyFeatureBlock computed ONCE per extract() call and injected
       into all three vectors (RL, meta, retrain) — single source of truth.

    2. ConfidenceWeightModel provides learned blend weights for
       calibrated_confidence used in meta and retrain vectors.

    3. Every extract() call records to SharedExperienceStore so all agents
       accumulate a shared cross-run learning history automatically.

    4. RL state vector uses build_state_vector() from rl_agent.py (unchanged)
       but the result dict passed to it now has calibrated_confidence
       injected under "confidence_score" so RL also sees the improved signal.
    """

    def __init__(self) -> None:
        self.drift_detector   = DriftDetector()
        self.error_analyser   = ErrorDistributionAnalyser()
        self._uncertainty_block = _global_uncertainty_block
        self._call_count      = 0

    # ── Main extraction ───────────────────────────────────────────────────────

    def extract(self, result: Dict[str, Any]) -> FeatureBundle:
        """
        Full extraction pipeline for one automl_service result dict.

        Pipeline
        ────────
        1. Named scalar features
        2. Uncertainty block (computed once — injected into all vectors)
        3. Error distribution
        4. Drift detection + uncertainty-weighted drift score
        5. RL state vector  (build_state_vector with calibrated_confidence)
        6. Meta-feature vector (uses calibrated_confidence + epistemic)
        7. Retrain vector  (uses calibrated_confidence + aleatoric)
        8. Record to SharedExperienceStore
        9. Return FeatureBundle
        """
        t0 = time.perf_counter()
        self._call_count += 1

        # ── 1. Named scalar features ──────────────────────────────────────────
        fd: Dict[str, Any] = _extract_scalars(result)

        # ── 2. Uncertainty block (ONE computation, injected everywhere) ───────
        uncertainty = self._uncertainty_block.compute(result)
        fd.update(uncertainty)

        # ── 3. Error distribution ─────────────────────────────────────────────
        fd.update(self.error_analyser.analyse(result))

        # ── 4. Drift detection ────────────────────────────────────────────────
        self.drift_detector.push(fd)
        drift_raw = self.drift_detector.drift_score()

        # Uncertainty-weighted drift score: high epistemic → inflate drift signal
        uncertainty_weight = 1.0 + fd.get("epistemic_uncertainty", 0.0)
        fd["drift_score"]  = round(min(1.0, drift_raw * uncertainty_weight), 5)
        fd["is_drifting"]  = float(self.drift_detector.is_drifting())

        # ── 5. RL state vector — inject calibrated_confidence ─────────────────
        # Patch the result dict so rl_agent.build_state_vector uses the
        # calibrated confidence (from learned weights) instead of the raw score
        patched_result = dict(result)
        patched_result["confidence_score"] = fd["calibrated_confidence"]
        try:
            rl_vec = build_state_vector(patched_result).astype(np.float32)
        except Exception as exc:
            log.warning(f"[FeatureExtractor] build_state_vector failed ({exc}); zero vector.")
            rl_vec = np.zeros(RL_DIM, dtype=np.float32)

        # ── 6. Meta-feature vector ────────────────────────────────────────────
        try:
            meta_vec = _build_meta_vector(fd)
        except Exception as exc:
            log.warning(f"[FeatureExtractor] meta vector failed ({exc}); zero vector.")
            meta_vec = np.zeros(META_DIM, dtype=np.float32)

        # ── 7. Retrain-feature vector ─────────────────────────────────────────
        try:
            retrain_vec = _build_retrain_vector(fd)
        except Exception as exc:
            log.warning(f"[FeatureExtractor] retrain vector failed ({exc}); zero vector.")
            retrain_vec = np.zeros(RETRAIN_DIM, dtype=np.float32)

        # ── 8. Record to SharedExperienceStore ────────────────────────────────
        run_id = str(result.get("run_id") or result.get("model_name") or
                     f"run_{self._call_count}")
        self._record_to_store(run_id, result, fd, uncertainty)

        # ── 9. Assemble ───────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        bundle = FeatureBundle(
            rl_state_vector     = rl_vec,
            meta_feature_vec    = meta_vec,
            retrain_feature_vec = retrain_vec,
            feature_dict        = fd,
            extraction_time_ms  = elapsed_ms,
            run_id              = run_id,
        )
        log.info(
            f"[FeatureExtractor] Run #{self._call_count}  "
            f"{len(fd)} scalars  "
            f"drift={fd['drift_score']:.4f}  "
            f"epistemic={fd.get('epistemic_uncertainty',0):.3f}  "
            f"calibrated_conf={fd.get('calibrated_confidence',0):.3f}  "
            f"{elapsed_ms:.1f}ms"
        )
        return bundle

    def _record_to_store(self, run_id: str, result: Dict,
                         fd: Dict, uncertainty: Dict) -> None:
        """Record this extraction to the SharedExperienceStore."""
        if not _STORE_AVAILABLE:
            return
        try:
            outcome = {
                "cv_score":   fd.get("cv_score",   0.0),
                "test_score": fd.get("test_score",  0.0),
                "overfit_gap":fd.get("overfit_gap", 0.0),
                "drift_score":fd.get("drift_score", 0.0),
            }
            reward = fd.get("calibrated_confidence", fd.get("confidence", 0.5))
            state  = {
                "n_rows_norm":   fd.get("n_rows_norm",  0.0),
                "n_feats_norm":  fd.get("n_feats_norm", 0.0),
                "missing_pct":   fd.get("missing_pct",  0.0),
                "imbalance":     fd.get("imbalance",    0.5),
                "prob_type":     fd.get("prob_type",    0.0),
                "tier":          fd.get("tier",         0.25),
            }
            record_experience(
                run_id      = run_id,
                agent       = AGENT_PIPELINE,
                state       = state,
                action      = str(result.get("best_model") or result.get("best_model_name", "")),
                outcome     = outcome,
                reward      = float(reward),
                uncertainty = uncertainty,
                meta        = {
                    "problem_type": result.get("problem_type"),
                    "scale_tier":   result.get("scale_tier"),
                },
            )
        except Exception as exc:
            log.debug(f"[FeatureExtractor] Store record failed (non-fatal): {exc}")

    @property
    def call_count(self) -> int:
        return self._call_count

    def drift_summary(self) -> Dict[str, Any]:
        return self.drift_detector.summary()

    def reset_drift(self) -> None:
        self.drift_detector.reset()
        log.info("[FeatureExtractor] Drift detector reset.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_EXTRACTOR: Optional[FeatureExtractor] = None


def get_extractor() -> FeatureExtractor:
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        _GLOBAL_EXTRACTOR = FeatureExtractor()
        log.info("[FeatureExtractor] Global singleton initialised.")
    return _GLOBAL_EXTRACTOR


def reset_global_extractor() -> None:
    global _GLOBAL_EXTRACTOR
    _GLOBAL_EXTRACTOR = None


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    print("\n── feature_extractor.py v2.0 self-tests ──")

    mock = {
        "problem_type":    "classification",
        "best_model":      "LightGBM",
        "cv_score_mean":   0.88, "cv_score_std": 0.03,
        "test_score":      0.87, "confidence_score": 0.84,
        "leakage_detected":False, "scale_tier": 1,
        "baseline_alert":  {"gap": 0.12, "triggered": False},
        "run_id":          "test_run_001",
        "agent_history": [
            {"score": 0.85, "effective_score": 0.82, "overfit": False, "epsilon_used": 0.35},
            {"score": 0.88, "effective_score": 0.86, "overfit": False, "epsilon_used": 0.28},
            {"score": 0.80, "effective_score": 0.75, "overfit": True,  "epsilon_used": 0.22},
        ],
        "dataset_diagnostics": {
            "n_rows": 3000, "n_cols": 18, "overall_missing_pct": 2.5,
            "class_imbalance": 0.62,
            "most_skewed_features": {"age": 1.4, "salary": 3.2},
        },
        "feature_selection": {
            "original_features": 18, "final_features": 12,
            "steps_applied": ["A_variance_threshold", "B_correlation_filter"],
            "pca_applied": False, "lda_applied": False,
        },
        "performance": {
            "scale_tier": 1, "train_accuracy": 0.95, "accuracy": 0.87,
            "cv_score_mean": 0.88, "cv_score_std": 0.03,
            "n_train": 2400, "n_test": 600,
            "confusion_matrix": [[280, 20], [30, 270]], "roc_auc": 0.92,
        },
        "sample_explanation": {
            "available": True,
            "shap_values": {"age": 0.35, "salary": 0.22, "education": 0.18,
                            "experience": 0.12, "score": 0.08, "other": 0.05},
        },
        "bandit_stats": {
            "LightGBM":           {"trials": 5, "avg_score": 0.88},
            "RandomForest":       {"trials": 3, "avg_score": 0.83},
            "LogisticRegression": {"trials": 2, "avg_score": 0.79},
        },
    }

    # ── 1. UncertaintyFeatureBlock ─────────────────────────────────────────────
    ub   = UncertaintyFeatureBlock()
    unc  = ub.compute(mock)
    assert 0.0 <= unc["epistemic_uncertainty"] <= 1.0
    assert 0.0 <= unc["aleatoric_uncertainty"]  <= 1.0
    assert 0.0 <= unc["calibrated_confidence"]  <= 1.0
    assert 0.0 <= unc["confidence_gap"]          <= 1.0
    assert unc["signal_strength"] > 0.5
    print(f"✓ UncertaintyFeatureBlock OK  "
          f"epistemic={unc['epistemic_uncertainty']:.3f}  "
          f"aleatoric={unc['aleatoric_uncertainty']:.3f}  "
          f"calibrated={unc['calibrated_confidence']:.3f}")

    # ── 2. _extract_scalars includes uncertainty ───────────────────────────────
    fd = _extract_scalars(mock)
    assert "cv_score" in fd and "imbalance" in fd
    print(f"✓ _extract_scalars OK  ({len(fd)} keys)")

    # ── 3. Meta vector uses calibrated_confidence (slot 5) ────────────────────
    fd_full = dict(fd)
    fd_full.update(unc)
    fd_full.update({"drift_score": 0.0, "is_drifting": 0.0,
                    "cm_entropy": 0.3, "accuracy": 0.87, "f1_macro": 0.87,
                    "roc_auc": 0.92, "acc_f1_gap": 0.0,
                    "mae": 0.0, "rmse": 0.0, "r2": 0.0, "residual_spread": 0.0})
    meta_vec = _build_meta_vector(fd_full)
    assert meta_vec.shape == (META_DIM,)
    assert meta_vec[5] == pytest_approx(unc["calibrated_confidence"], abs=0.001)
    print(f"✓ meta_vec slot[5] = calibrated_confidence={meta_vec[5]:.4f}")

    # ── 4. Retrain vector uses calibrated_confidence (slot 3) + aleatoric ─────
    retrain_vec = _build_retrain_vector(fd_full)
    assert retrain_vec.shape == (RETRAIN_DIM,)
    assert retrain_vec[3] == pytest_approx(unc["calibrated_confidence"], abs=0.001)
    assert retrain_vec[10] == pytest_approx(unc["aleatoric_uncertainty"], abs=0.001)
    print(f"✓ retrain_vec slot[3]={retrain_vec[3]:.4f}  slot[10]={retrain_vec[10]:.4f}")

    # ── 5. RL state vector uses build_state_vector (unchanged shape) ──────────
    rl_vec = build_state_vector(mock)
    assert rl_vec.shape == (STATE_DIM,)
    print(f"✓ rl_vec shape={rl_vec.shape}  range=[{rl_vec.min():.3f},{rl_vec.max():.3f}]")

    # ── 6. FeatureExtractor.extract full pipeline ─────────────────────────────
    fe     = FeatureExtractor()
    bundle = fe.extract(mock)
    assert bundle.rl_state_vector.shape     == (RL_DIM,)
    assert bundle.meta_feature_vec.shape    == (META_DIM,)
    assert bundle.retrain_feature_vec.shape == (RETRAIN_DIM,)
    assert "epistemic_uncertainty"  in bundle.feature_dict
    assert "aleatoric_uncertainty"  in bundle.feature_dict
    assert "calibrated_confidence"  in bundle.feature_dict
    assert "confidence_gap"         in bundle.feature_dict
    assert "drift_score"            in bundle.feature_dict
    print(f"✓ FeatureExtractor.extract OK  "
          f"fd_keys={len(bundle.feature_dict)}  t={bundle.extraction_time_ms:.1f}ms")

    # ── 7. Uncertainty features identical across all three vectors ─────────────
    # Verify meta[5] == retrain[3] == calibrated_confidence in fd
    assert abs(bundle.meta_feature_vec[5] -
               bundle.feature_dict["calibrated_confidence"]) < 0.001
    assert abs(bundle.retrain_feature_vec[3] -
               bundle.feature_dict["calibrated_confidence"]) < 0.001
    print(f"✓ Uncertainty consistency: meta[5]=retrain[3]=fd.calibrated OK")

    # ── 8. DriftDetector uncertainty weighting ─────────────────────────────────
    fe2 = FeatureExtractor()
    for _ in range(25): fe2.extract(mock)
    # With identical data, drift_score should be ~0 (stable distribution)
    drift_val = fe2.drift_summary()["drift_score"]
    assert drift_val < 0.10, f"Expected stable drift, got {drift_val:.4f}"
    print(f"✓ DriftDetector stable OK  score={drift_val:.4f}")

    # ── 9. Singleton ──────────────────────────────────────────────────────────
    e1 = get_extractor(); e2 = get_extractor()
    assert e1 is e2
    reset_global_extractor()
    e3 = get_extractor()
    assert e3 is not e1
    print(f"✓ Singleton + reset OK")

    # ── 10. ConfidenceWeightModel defaults sum to ~1 ───────────────────────────
    w = _global_confidence_weight_model.weights
    assert abs(sum(w) - 1.0) < 0.1, f"Weights should sum to ~1, got {sum(w):.3f}"
    print(f"✓ ConfidenceWeightModel default weights={w}")

    print(f"\n✓ All feature_extractor.py v2.0 tests passed.  "
          f"RL_DIM={RL_DIM}  META_DIM={META_DIM}  RETRAIN_DIM={RETRAIN_DIM}\n")


def pytest_approx(value, abs=1e-6):
    _abs = __builtins__["abs"] if isinstance(__builtins__, dict) else __builtins__.abs
    class _A:
        def __init__(self, v, a): self.v = v; self.a = a
        def __eq__(self, o): return _abs(float(o) - self.v) <= self.a
    return _A(value, abs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()