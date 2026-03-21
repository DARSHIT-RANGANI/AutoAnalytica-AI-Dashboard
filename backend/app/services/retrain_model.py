"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  retrain_model.py  — v2.0  (AutoAnalytica v5.5)                            ║
║                                                                              ║
║  Drift Detection & Retrain Decision Engine                                   ║
║                                                                              ║
║  v2.0 Changes                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  + RetrainScorer       : GBM classifier replaces MEDIUM×2 / LOW×3 rules    ║
║  + AdaptiveThresholdLearner : Ridge learns drop/variance thresholds         ║
║  + record() triggers incremental fit of both learned models                 ║
║  + _aggregate_alerts() is model-first, rules only for CRITICAL/HIGH         ║
║    and catastrophic failures (safety net)                                    ║
║                                                                              ║
║  Only HARD rule kept                                                         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  CATASTROPHIC_FAILURE: if CV score drops below 0.30 absolute, or            ║
║  leakage is detected → immediate retrain regardless of model confidence.    ║
║                                                                              ║
║  Persistence                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  retrain_state/run_ledger.json    — rolling window of per-run snapshots     ║
║  retrain_state/retrain_meta.json  — controller config + retrain history     ║
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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_RETRAIN_STATE_DIR = Path("retrain_state")
_RUN_LEDGER_PATH   = _RETRAIN_STATE_DIR / "run_ledger.json"
_RETRAIN_META_PATH = _RETRAIN_STATE_DIR / "retrain_meta.json"

_RETRAIN_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY LEVELS  (kept for alert taxonomy — NOT for decision gating)
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH     = "HIGH"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_LOW      = "LOW"

# ── Safety-only thresholds (catastrophic failure gate — NOT learnable) ───────
_CATASTROPHIC_CV_FLOOR = 0.30  # absolute CV score below this → always retrain
_MEDIUM_THRESHOLD = 2          # fallback only — used when RetrainScorer absent
_LOW_THRESHOLD    = 3          # fallback only

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "drop_threshold":           0.05,
    "variance_threshold":       0.08,
    "overfit_streak_n":         3,
    "confidence_streak_n":      3,
    "volume_shift_pct":         30.0,
    "missing_spike_pp":         10.0,
    "max_days_without_retrain": 30,
    "model_flip_streak_n":      4,
    "window_size":              20,
}


# ─────────────────────────────────────────────────────────────────────────────
# ALERT  (single trigger result — unchanged from v1.0)
# ─────────────────────────────────────────────────────────────────────────────

class Alert:
    def __init__(self, trigger, severity, fired, message, values=None):
        self.trigger  = trigger
        self.severity = severity
        self.fired    = fired
        self.message  = message
        self.values   = values or {}

    def to_dict(self):
        return {"trigger": self.trigger, "severity": self.severity,
                "fired": self.fired, "message": self.message, "values": self.values}

    def __repr__(self):
        icon = "🔴" if self.fired else "🟢"
        return f"{icon} [{self.severity}] {self.trigger}: {self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# RUN SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def _extract_snapshot(pipeline_result: Dict) -> Dict:
    perf = pipeline_result.get("performance") or pipeline_result.get("metrics") or {}
    diag = (pipeline_result.get("dataset_diagnostics")
            or pipeline_result.get("intermediate_outputs", {})
                               .get("transform", {}).get("diagnostics") or {})
    fs   = (perf.get("feature_selection") or pipeline_result.get("feature_selection") or {})

    def _sf(key, *fallbacks, default=None):
        for k in (key, *fallbacks):
            v = perf.get(k) or pipeline_result.get(k) or diag.get(k)
            if v is not None:
                try:
                    fv = float(v)
                    return None if (math.isnan(fv) or math.isinf(fv)) else fv
                except (TypeError, ValueError): pass
        return default

    def _sb(key, *fallbacks):
        for k in (key, *fallbacks):
            v = perf.get(k) or pipeline_result.get(k) or diag.get(k)
            if v is not None: return bool(v)
        return False

    return {
        "ts":                  _ts(),
        "cv_mean":             _sf("cv_score_mean"),
        "cv_std":              _sf("cv_score_std"),
        "test_score":          _sf("test_score", "accuracy", "R2"),
        "confidence_score":    _sf("confidence_score"),
        "confidence_label":    perf.get("confidence_label") or pipeline_result.get("confidence_label"),
        "overfitting":         _sb("overfitting"),
        "leakage_detected":    _sb("leakage_detected"),
        "baseline_alert":      bool((pipeline_result.get("baseline_alert") or {}).get("triggered", False)),
        "n_rows":              _sf("n_rows", "rows"),
        "n_cols":              _sf("n_cols", "features"),
        "overall_missing_pct": _sf("overall_missing_pct"),
        "n_classes":           _sf("n_classes"),
        "imbalance_ratio":     _sf("imbalance_ratio"),
        "orig_features":       float(fs.get("original_features") or _sf("original_features") or 0),
        "best_model":          pipeline_result.get("best_model_name") or pipeline_result.get("best_model", "unknown"),
        "problem_type":        pipeline_result.get("problem_type", "unknown"),
        "scale_tier":          _sf("scale_tier"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN LEDGER
# ─────────────────────────────────────────────────────────────────────────────

class RunLedger:
    def __init__(self, records=None, max_size=200):
        self._records: List[Dict] = records or []
        self.max_size = max_size

    def append(self, snapshot):
        self._records.append(snapshot)
        if len(self._records) > self.max_size:
            self._records = self._records[-self.max_size:]

    def __len__(self): return len(self._records)
    def last(self, n=1): return self._records[-n:] if self._records else []

    def last_n_values(self, key, n):
        vals = []
        for r in reversed(self._records):
            v = r.get(key)
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv): vals.append(fv)
                except (TypeError, ValueError): pass
            if len(vals) >= n: break
        return list(reversed(vals))

    def rolling_mean(self, key, n):
        vals = self.last_n_values(key, n)
        return statistics.mean(vals) if vals else None

    def rolling_std(self, key, n):
        vals = self.last_n_values(key, n)
        return statistics.stdev(vals) if len(vals) >= 2 else None

    def baseline(self, key, window):
        vals = self.last_n_values(key, window)
        if not vals: return None
        older = vals[:max(1, len(vals) // 2)]
        return statistics.mean(older)

    def consecutive(self, key, predicate, n):
        vals = self.last_n_values(key, n)
        if len(vals) < n: return False, len(vals)
        streak = sum(1 for v in vals if predicate(v))
        return streak >= n, streak

    def last_str_values(self, key, n):
        vals = []
        for r in reversed(self._records):
            v = r.get(key)
            if v is not None: vals.append(str(v))
            if len(vals) >= n: break
        return list(reversed(vals))

    def to_dict(self):   return {"records": self._records, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, data):
        return cls(records=data.get("records", []), max_size=data.get("max_size", 200))


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN SCORER  (v2.0 — replaces MEDIUM×2 / LOW×3 aggregation)
# ─────────────────────────────────────────────────────────────────────────────

class RetrainScorer:
    """
    GradientBoosting classifier that learns whether a given combination of
    fired alerts should trigger a retrain — replacing the hardcoded
    _MEDIUM_THRESHOLD=2 / _LOW_THRESHOLD=3 counting logic.

    Training signal
    ───────────────
    Every retrain event recorded via mark_retrained() is labelled 1 (retrain
    warranted); runs between retrains provide 0 labels.
    Feature vector = 10 binary alert flags + 4 severity-count aggregates
    (14 features total).

    Safety design
    ─────────────
    - CRITICAL and HIGH alerts still fire unconditionally (hard rules — safety)
    - Only MEDIUM and LOW gating is replaced by the learned model
    - Falls back to count-based rules when insufficient history
    - CONFIDENCE_FLOOR=0.55 — model must be fairly confident to override rules
    """

    MIN_SAMPLES      = 25
    RETRAIN_EVERY    = 10
    CONFIDENCE_FLOOR = 0.55

    def __init__(self):
        self._clf          = None
        self._next_retrain = self.MIN_SAMPLES

    def maybe_fit(self, retrain_history: List[dict], ledger: RunLedger) -> None:
        if len(ledger) >= self._next_retrain:
            self._fit(retrain_history, ledger)
            self._next_retrain = len(ledger) + self.RETRAIN_EVERY

    def predict(self, alerts: List[Alert]) -> Optional[Tuple[bool, float]]:
        """
        Returns (should_retrain: bool, probability: float) or None.
        None means the model is absent / not confident → caller uses rule fallback.
        """
        if self._clf is None:
            return None
        try:
            vec    = self._alerts_to_vec(alerts)
            proba  = self._clf.predict_proba([vec])[0]
            classes = list(self._clf.classes_)
            if 1 not in classes:
                return None
            p_retrain = float(proba[classes.index(1)])
            # Only act when model is sufficiently confident
            if p_retrain < self.CONFIDENCE_FLOOR and (1 - p_retrain) < self.CONFIDENCE_FLOOR:
                return None
            return p_retrain >= self.CONFIDENCE_FLOOR, p_retrain
        except Exception as exc:
            log.warning(f"[RetrainScorer] predict failed: {exc}")
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _alerts_to_vec(self, alerts: List[Alert]) -> List[float]:
        """14-dim feature vector from alert list."""
        trigger_order = [
            "LEAKAGE_DETECTED", "PERFORMANCE_DROP", "DATA_SCHEMA_DRIFT",
            "HIGH_VARIANCE", "OVERFIT_STREAK", "DATA_VOLUME_SHIFT",
            "MISSING_RATE_SPIKE", "CONFIDENCE_LOW", "TIME_ELAPSED",
            "MODEL_FLIP_STREAK",
        ]
        alert_map = {a.trigger: a for a in alerts}
        vec = [float(alert_map[t].fired) if t in alert_map else 0.0
               for t in trigger_order]
        # Aggregate counts
        vec.append(float(sum(1 for a in alerts if a.fired)))
        vec.append(float(sum(1 for a in alerts if a.fired and a.severity == SEVERITY_CRITICAL)))
        vec.append(float(sum(1 for a in alerts if a.fired and a.severity == SEVERITY_HIGH)))
        vec.append(float(sum(1 for a in alerts if a.fired and a.severity == SEVERITY_MEDIUM)))
        return vec

    def _fit(self, retrain_history: List[dict], ledger: RunLedger) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            log.warning("[RetrainScorer] sklearn unavailable — skipping fit.")
            return

        retrain_ts_set = {r.get("ts") for r in retrain_history if r.get("ts")}
        records = ledger._records

        X: List[List[float]] = []
        y: List[int]          = []
        for i, rec in enumerate(records):
            label = 1 if (i < len(records) - 1 and
                          records[i + 1].get("ts") in retrain_ts_set) else 0
            vec = self._snapshot_to_mock_vec(rec)
            if vec is not None:
                X.append(vec)
                y.append(label)

        if sum(y) < 3 or (len(y) - sum(y)) < 5:
            log.debug("[RetrainScorer] Insufficient retrain labels — skipping fit.")
            return

        clf = GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42,
            min_samples_leaf=2)
        clf.fit(X, y)
        self._clf = clf
        log.info(f"[RetrainScorer] Fitted — {len(X)} samples, {sum(y)} retrain labels.")

    def _snapshot_to_mock_vec(self, rec: dict) -> Optional[List[float]]:
        """Derive coarse 14-dim alert vector from a ledger snapshot."""
        try:
            cv    = float(rec.get("cv_mean",  0.5) or 0.5)
            std   = float(rec.get("cv_std",   0.05) or 0.05)
            overf = float(bool(rec.get("overfitting",      False)))
            leak  = float(bool(rec.get("leakage_detected", False)))
            conf  = 1.0 if rec.get("confidence_label") == "Low" else 0.0
            # 10 trigger slots (binary)
            return [
                leak,                       # LEAKAGE_DETECTED
                float(cv < 0.70),           # PERFORMANCE_DROP proxy
                0.0,                        # DATA_SCHEMA_DRIFT (not in snapshot)
                float(std > 0.08),          # HIGH_VARIANCE
                overf,                      # OVERFIT_STREAK proxy
                0.0, 0.0,                   # vol / missing
                conf,                       # CONFIDENCE_LOW
                0.0, 0.0,                   # time / flip
                # 4 aggregate counts
                leak + float(cv < 0.70) + float(std > 0.08) + overf + conf,
                leak,
                float(cv < 0.70),
                float(std > 0.08) + overf + conf,
            ]
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLD LEARNER  (v2.0 — replaces hardcoded DEFAULTS)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveThresholdLearner:
    """
    Ridge regression that learns where each numeric threshold should sit
    by finding the decision boundary that best predicts retrain events.

    Replaces the static initial values in DEFAULTS for:
      - drop_threshold:     where performance drop triggers concern
      - variance_threshold: where CV std triggers concern

    Algorithm
    ─────────
    For each threshold key, fit Ridge(y=retrain_label) on the raw metric
    values. Solve for the boundary x where predicted P(retrain)=0.5.
    Clamp to safety bounds so thresholds never drift to dangerous extremes.
    """

    MIN_SAMPLES = 30
    RETRAIN_EVERY = 10

    _LEARNABLE = {
        "drop_threshold":     (0.02, 0.15),
        "variance_threshold": (0.04, 0.20),
    }

    def __init__(self):
        self._learned: Dict[str, float] = {}
        self._next_retrain = self.MIN_SAMPLES

    def get(self, key: str, default: float) -> float:
        return self._learned.get(key, default)

    def maybe_fit(self, retrain_history: List[dict], ledger: RunLedger) -> None:
        if len(ledger) >= self._next_retrain and retrain_history:
            self._fit(retrain_history, ledger)
            self._next_retrain = len(ledger) + self.RETRAIN_EVERY

    def _fit(self, retrain_history: List[dict], ledger: RunLedger) -> None:
        try:
            from sklearn.linear_model import Ridge
            import numpy as np
        except ImportError:
            return

        retrain_ts_set = {r.get("ts") for r in retrain_history if r.get("ts")}
        records = ledger._records

        X_drop, X_var, y = [], [], []
        for i, rec in enumerate(records):
            cv  = rec.get("cv_mean"); std = rec.get("cv_std")
            if cv is None or std is None: continue
            label = 1 if (i < len(records) - 1 and
                          records[i + 1].get("ts") in retrain_ts_set) else 0
            X_drop.append([float(cv)])
            X_var.append([float(std)])
            y.append(label)

        if sum(y) < 3: return
        y_arr = np.array(y)

        for key, feat_list, (lo, hi) in [
            ("drop_threshold",     X_drop, self._LEARNABLE["drop_threshold"]),
            ("variance_threshold", X_var,  self._LEARNABLE["variance_threshold"]),
        ]:
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(feat_list, y_arr)
                coef = float(reg.coef_[0]); intercept = float(reg.intercept_)
                if abs(coef) > 1e-6:
                    boundary = (0.5 - intercept) / coef
                    self._learned[key] = float(max(lo, min(hi, boundary)))
            except Exception: pass

        if self._learned:
            log.info(f"[AdaptiveThresholdLearner] Updated: {self._learned}")


# Module-level singletons
_global_retrain_scorer    = RetrainScorer()
_global_threshold_learner = AdaptiveThresholdLearner()


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL DRIFT DETECTORS  (unchanged from v1.0 — detectors are observation
# logic, not decision logic; the decision logic moves to RetrainScorer)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_leakage(snapshot, **_):
    fired = bool(snapshot.get("leakage_detected"))
    return Alert("LEAKAGE_DETECTED", SEVERITY_CRITICAL, fired,
                 "Data leakage detected — retrain mandatory." if fired
                 else "No leakage.", {"leakage_detected": fired})


def _detect_performance_drop(snapshot, ledger, drop_threshold, window_size, **_):
    cv_now  = snapshot.get("cv_mean")
    cv_base = ledger.baseline("cv_mean", window_size)
    if cv_now is None or cv_base is None:
        return Alert("PERFORMANCE_DROP", SEVERITY_HIGH, False,
                     "Insufficient history.", {"cv_now": cv_now})
    drop  = cv_base - cv_now
    fired = drop > drop_threshold
    return Alert("PERFORMANCE_DROP", SEVERITY_HIGH, fired,
                 (f"CV dropped {drop*100:.1f}pp (base={cv_base:.4f}, now={cv_now:.4f})."
                  if fired else f"CV stable (Δ={drop*100:.1f}pp)."),
                 {"cv_now": round(cv_now,4), "cv_baseline": round(cv_base,4),
                  "drop": round(drop,4), "threshold": drop_threshold})


def _detect_schema_drift(snapshot, ledger, window_size, **_):
    n_cols_now     = snapshot.get("n_cols")
    n_classes_now  = snapshot.get("n_classes")
    n_cols_base    = ledger.baseline("n_cols",    window_size)
    n_classes_base = ledger.baseline("n_classes", window_size)
    cols_drifted    = (n_cols_now is not None and n_cols_base is not None
                       and abs(n_cols_now - n_cols_base) > 0)
    classes_drifted = (n_classes_now is not None and n_classes_base is not None
                       and abs(n_classes_now - n_classes_base) > 0)
    fired   = cols_drifted or classes_drifted
    details = []
    if cols_drifted:    details.append(f"n_cols {n_cols_base:.0f}→{n_cols_now:.0f}")
    if classes_drifted: details.append(f"n_classes {n_classes_base:.0f}→{n_classes_now:.0f}")
    return Alert("DATA_SCHEMA_DRIFT", SEVERITY_HIGH, fired,
                 (f"Schema changed: {', '.join(details)}." if fired
                  else "No schema drift."),
                 {"n_cols_now": n_cols_now, "n_cols_base": n_cols_base})


def _detect_high_variance(snapshot, ledger, variance_threshold, window_size, **_):
    cv_std_now  = snapshot.get("cv_std")
    cv_std_roll = ledger.rolling_mean("cv_std", window_size)
    ref   = cv_std_roll if cv_std_roll is not None else cv_std_now
    fired = ref is not None and ref > variance_threshold * 2
    return Alert("HIGH_VARIANCE", SEVERITY_MEDIUM, fired,
                 (f"Persistently high variance (rolling={ref:.4f} > {variance_threshold*2:.4f})."
                  if fired else f"Variance acceptable (rolling={ref})."),
                 {"cv_std_rolling": cv_std_roll, "threshold_x2": variance_threshold*2})


def _detect_overfit_streak(snapshot, ledger, overfit_streak_n, **_):
    recent  = ledger.last_n_values("overfitting", overfit_streak_n - 1)
    current = 1.0 if snapshot.get("overfitting") else 0.0
    combined = recent + [current]
    streak   = int(sum(1 for v in combined if v > 0.5))
    fired    = streak >= overfit_streak_n
    return Alert("OVERFIT_STREAK", SEVERITY_MEDIUM, fired,
                 f"Overfitting in {streak}/{overfit_streak_n} recent runs.",
                 {"streak": streak, "required": overfit_streak_n})


def _detect_volume_shift(snapshot, ledger, volume_shift_pct, window_size, **_):
    n_rows_now  = snapshot.get("n_rows")
    n_rows_base = ledger.baseline("n_rows", window_size)
    if n_rows_now is None or n_rows_base is None or n_rows_base == 0:
        return Alert("DATA_VOLUME_SHIFT", SEVERITY_MEDIUM, False,
                     "Insufficient history.", {"n_rows_now": n_rows_now})
    pct_change = abs(n_rows_now - n_rows_base) / n_rows_base * 100
    fired = pct_change > volume_shift_pct
    return Alert("DATA_VOLUME_SHIFT", SEVERITY_MEDIUM, fired,
                 (f"Row count Δ={pct_change:.1f}% (base={n_rows_base:.0f}, now={n_rows_now:.0f})."
                  if fired else f"Volume stable (Δ={pct_change:.1f}%)."),
                 {"n_rows_now": n_rows_now, "pct_change": round(pct_change,2)})


def _detect_missing_spike(snapshot, ledger, missing_spike_pp, window_size, **_):
    miss_now  = snapshot.get("overall_missing_pct")
    miss_base = ledger.baseline("overall_missing_pct", window_size)
    if miss_now is None or miss_base is None:
        return Alert("MISSING_RATE_SPIKE", SEVERITY_MEDIUM, False,
                     "Insufficient history.", {"miss_now": miss_now})
    delta = miss_now - miss_base
    fired = delta > missing_spike_pp
    return Alert("MISSING_RATE_SPIKE", SEVERITY_MEDIUM, fired,
                 (f"Missing rate +{delta:.1f}pp (base={miss_base:.1f}%, now={miss_now:.1f}%)."
                  if fired else f"Missing stable (Δ={delta:+.1f}pp)."),
                 {"miss_now": round(miss_now,2), "delta_pp": round(delta,2)})


def _detect_confidence_streak(snapshot, ledger, confidence_streak_n, **_):
    recent  = ledger.last_str_values("confidence_label", confidence_streak_n - 1)
    current = snapshot.get("confidence_label", "")
    combined = recent + [current]
    low_count = sum(1 for v in combined if v == "Low")
    fired = low_count >= confidence_streak_n
    return Alert("CONFIDENCE_LOW", SEVERITY_MEDIUM, fired,
                 f"Confidence 'Low' in {low_count}/{confidence_streak_n} recent runs.",
                 {"low_count": low_count, "required": confidence_streak_n})


def _detect_time_elapsed(snapshot, ledger, max_days_without_retrain,
                          last_retrain_ts, **_):
    if last_retrain_ts is None:
        records = ledger.last(len(ledger))
        first_ts = records[0].get("ts") if records else None
    else:
        first_ts = last_retrain_ts
    if first_ts is None:
        return Alert("TIME_ELAPSED", SEVERITY_LOW, False,
                     "No retrain timestamp.", {"last_retrain_ts": None})
    try:
        ref_dt = datetime.fromisoformat(first_ts)
        now_dt = datetime.now(timezone.utc)
        if ref_dt.tzinfo is None: ref_dt = ref_dt.replace(tzinfo=timezone.utc)
        elapsed_days = (now_dt - ref_dt).days
    except Exception as exc:
        return Alert("TIME_ELAPSED", SEVERITY_LOW, False,
                     f"Could not parse ts: {exc}", {})
    fired = elapsed_days > max_days_without_retrain
    return Alert("TIME_ELAPSED", SEVERITY_LOW, fired,
                 f"{elapsed_days}d since last retrain (threshold={max_days_without_retrain}d).",
                 {"elapsed_days": elapsed_days, "threshold_days": max_days_without_retrain})


def _detect_model_flip_streak(snapshot, ledger, model_flip_streak_n, **_):
    recent  = ledger.last_str_values("best_model", model_flip_streak_n - 1)
    current = str(snapshot.get("best_model", ""))
    combined = recent + [current]
    if len(combined) < model_flip_streak_n:
        return Alert("MODEL_FLIP_STREAK", SEVERITY_LOW, False,
                     "Insufficient history.", {})
    unique_models = len(set(combined))
    fired = unique_models == len(combined)
    return Alert("MODEL_FLIP_STREAK", SEVERITY_LOW, fired,
                 (f"Model changed every run over last {len(combined)} runs — instability."
                  if fired else f"Model stable ({unique_models} unique in last {len(combined)})."),
                 {"models": combined, "unique_models": unique_models})


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_DETECTORS = [
    _detect_leakage,           # CRITICAL
    _detect_performance_drop,  # HIGH
    _detect_schema_drift,      # HIGH
    _detect_high_variance,     # MEDIUM
    _detect_overfit_streak,    # MEDIUM
    _detect_volume_shift,      # MEDIUM
    _detect_missing_spike,     # MEDIUM
    _detect_confidence_streak, # MEDIUM
    _detect_time_elapsed,      # LOW
    _detect_model_flip_streak, # LOW
]


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION LOGIC  (v2.0 — model-first)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_alerts(alerts: List[Alert],
                      snapshot: Optional[Dict] = None) -> Tuple[bool, str, str]:
    """
    v2.0 — Model-first alert aggregation.

    Decision hierarchy
    ──────────────────
    1. CATASTROPHIC FAILURE (hard rule — safety)
       CV score < 0.30 absolute OR leakage detected → always retrain
       These bypass everything else — they are system safety conditions.

    2. CRITICAL triggers (hard rule — kept)
       Any CRITICAL alert fires → retrain immediately.

    3. HIGH triggers (hard rule — kept)
       Any HIGH alert fires → retrain immediately.

    4. MEDIUM + LOW (learned model)
       RetrainScorer predicts whether the fired combination warrants retraining.
       Falls back to count thresholds when model is absent or low-confidence.

    Why keep CRITICAL/HIGH as rules?
    ─────────────────────────────────
    CRITICAL = data leakage (integrity issue — always retrain)
    HIGH     = measurable performance drop or schema change (high-confidence signal)
    These are direct, easily observable evidence — not ambiguous patterns
    that benefit from learned weighting.
    """
    fired = [a for a in alerts if a.fired]

    if not fired:
        return False, "All drift detectors clear — no retrain needed.", "NONE"

    # ── 1. CATASTROPHIC FAILURE (absolute safety gate) ─────────────────────────
    if snapshot is not None:
        cv_now = snapshot.get("cv_mean")
        if cv_now is not None and float(cv_now) < _CATASTROPHIC_CV_FLOOR:
            return (True,
                    f"CATASTROPHIC: CV score {cv_now:.4f} below floor {_CATASTROPHIC_CV_FLOOR}.",
                    SEVERITY_CRITICAL)
        if snapshot.get("leakage_detected"):
            return (True, "CATASTROPHIC: Data leakage detected — mandatory retrain.",
                    SEVERITY_CRITICAL)

    # ── 2. CRITICAL (hard rule) ────────────────────────────────────────────────
    criticals = [a for a in fired if a.severity == SEVERITY_CRITICAL]
    if criticals:
        msgs = "; ".join(a.message for a in criticals)
        return True, f"CRITICAL trigger(s): {msgs}", SEVERITY_CRITICAL

    # ── 3. HIGH (hard rule) ────────────────────────────────────────────────────
    highs = [a for a in fired if a.severity == SEVERITY_HIGH]
    if highs:
        msgs = "; ".join(a.message for a in highs)
        return True, f"HIGH trigger(s): {msgs}", SEVERITY_HIGH

    # ── 4. MEDIUM + LOW (learned model — primary) ─────────────────────────────
    scorer_result = _global_retrain_scorer.predict(alerts)
    if scorer_result is not None:
        should, prob = scorer_result
        fired_names  = [a.trigger for a in fired]
        severity     = SEVERITY_MEDIUM if should else "MILD"
        reason = (f"RetrainScorer: p={prob:.3f} → {'retrain' if should else 'no retrain'} "
                  f"(fired: {fired_names})")
        log.info(f"[Retrain] Model decision: should={should}, p={prob:.3f}")
        return should, reason, severity

    # ── 5. Count-based fallback (active only when RetrainScorer absent) ────────
    mediums = [a for a in fired if a.severity == SEVERITY_MEDIUM]
    if len(mediums) >= _MEDIUM_THRESHOLD:
        msgs = "; ".join(a.message for a in mediums)
        return True, f"Multiple MEDIUM ({len(mediums)}): {msgs}", SEVERITY_MEDIUM

    lows = [a for a in fired if a.severity == SEVERITY_LOW]
    if len(lows) >= _LOW_THRESHOLD:
        msgs = "; ".join(a.message for a in lows)
        return True, f"Multiple LOW ({len(lows)}): {msgs}", SEVERITY_LOW

    # Below threshold
    msg_parts = []
    if mediums:
        msg_parts.append(f"{len(mediums)} MEDIUM (need {_MEDIUM_THRESHOLD}): "
                         + "; ".join(a.trigger for a in mediums))
    if lows:
        msg_parts.append(f"{len(lows)} LOW (need {_LOW_THRESHOLD}): "
                         + "; ".join(a.trigger for a in lows))
    return False, "Mild drift below threshold. " + " | ".join(msg_parts), "MILD"


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class RetrainController:
    """
    Stateful drift detector and retrain decision engine.

    v2.0: Wires RetrainScorer and AdaptiveThresholdLearner into the
    record() lifecycle so both models improve with every new pipeline run.
    """

    VERSION = "2.0"

    def __init__(self, ledger=None, config=None, retrain_history=None,
                 last_retrain_ts=None, total_runs=0):
        self.ledger          = ledger or RunLedger()
        self.config          = {**DEFAULTS, **(config or {})}
        self.retrain_history = retrain_history or []
        self.last_retrain_ts = last_retrain_ts
        self.total_runs      = total_runs

    @classmethod
    def load_or_create(cls) -> "RetrainController":
        ledger_data = _load_json(_RUN_LEDGER_PATH)
        meta_data   = _load_json(_RETRAIN_META_PATH)
        ledger = RunLedger.from_dict(ledger_data) if ledger_data else RunLedger()
        inst   = cls(
            ledger          = ledger,
            config          = meta_data.get("config", {}),
            retrain_history = meta_data.get("retrain_history", []),
            last_retrain_ts = meta_data.get("last_retrain_ts"),
            total_runs      = int(meta_data.get("total_runs", 0)),
        )
        # Warm up learned models from persisted history
        _global_retrain_scorer.maybe_fit(inst.retrain_history, inst.ledger)
        _global_threshold_learner.maybe_fit(inst.retrain_history, inst.ledger)
        log.info(f"[Retrain] Loaded — runs={inst.total_runs}, ledger={len(ledger)}")
        return inst

    # ── Core API ──────────────────────────────────────────────────────────────

    def should_retrain(self, pipeline_result: Dict) -> Tuple[bool, str]:
        """
        Run all drift detectors + learned scorer against the latest result.
        Returns (should_retrain: bool, reason: str).
        """
        snapshot = _extract_snapshot(pipeline_result)
        # Use learned thresholds where available
        cfg = dict(self.config)
        for key in ["drop_threshold", "variance_threshold"]:
            cfg[key] = _global_threshold_learner.get(key, cfg[key])

        kwargs = {
            "snapshot":                snapshot,
            "ledger":                  self.ledger,
            "last_retrain_ts":         self.last_retrain_ts,
            "drop_threshold":          cfg["drop_threshold"],
            "variance_threshold":      cfg["variance_threshold"],
            "overfit_streak_n":        cfg["overfit_streak_n"],
            "confidence_streak_n":     cfg["confidence_streak_n"],
            "volume_shift_pct":        cfg["volume_shift_pct"],
            "missing_spike_pp":        cfg["missing_spike_pp"],
            "max_days_without_retrain":cfg["max_days_without_retrain"],
            "model_flip_streak_n":     cfg["model_flip_streak_n"],
            "window_size":             cfg["window_size"],
        }
        alerts = [detector(**kwargs) for detector in _DETECTORS]
        should, reason, severity = _aggregate_alerts(alerts, snapshot=snapshot)

        fired_names = [a.trigger for a in alerts if a.fired]
        log.info(f"[Retrain] should={should} severity={severity} "
                 f"fired={fired_names} reason={reason[:80]}…")
        return should, reason

    def full_analysis(self, pipeline_result: Dict) -> Dict:
        """Full structured analysis for /retrain_status endpoint."""
        snapshot = _extract_snapshot(pipeline_result)
        cfg = dict(self.config)
        for key in ["drop_threshold", "variance_threshold"]:
            cfg[key] = _global_threshold_learner.get(key, cfg[key])

        kwargs = {
            "snapshot": snapshot, "ledger": self.ledger,
            "last_retrain_ts": self.last_retrain_ts,
            "drop_threshold": cfg["drop_threshold"],
            "variance_threshold": cfg["variance_threshold"],
            "overfit_streak_n": cfg["overfit_streak_n"],
            "confidence_streak_n": cfg["confidence_streak_n"],
            "volume_shift_pct": cfg["volume_shift_pct"],
            "missing_spike_pp": cfg["missing_spike_pp"],
            "max_days_without_retrain": cfg["max_days_without_retrain"],
            "model_flip_streak_n": cfg["model_flip_streak_n"],
            "window_size": cfg["window_size"],
        }
        alerts = [detector(**kwargs) for detector in _DETECTORS]
        should, reason, severity = _aggregate_alerts(alerts, snapshot=snapshot)

        return {
            "should_retrain":         should,
            "reason":                 reason,
            "severity":               severity,
            "fired_triggers":         [a.trigger for a in alerts if a.fired],
            "all_alerts":             [a.to_dict() for a in alerts],
            "snapshot":               snapshot,
            "total_runs":             self.total_runs,
            "ledger_size":            len(self.ledger),
            "last_retrain_ts":        self.last_retrain_ts,
            "config":                 cfg,
            "scorer_ready":           _global_retrain_scorer._clf is not None,
            "learned_thresholds":     _global_threshold_learner._learned,
        }

    def record(self, pipeline_result: Dict) -> None:
        """
        Append snapshot to ledger.  Call after every run.

        v2.0 — also triggers incremental re-fitting of RetrainScorer and
        AdaptiveThresholdLearner so decisions improve over time.
        """
        snap = _extract_snapshot(pipeline_result)
        self.ledger.append(snap)
        self.total_runs += 1

        # v2.0 — incremental learning
        _global_retrain_scorer.maybe_fit(self.retrain_history, self.ledger)
        _global_threshold_learner.maybe_fit(self.retrain_history, self.ledger)

        log.debug(f"[Retrain] Ledger updated — total={self.total_runs}")

    def mark_retrained(self, reason: str = "") -> None:
        ts = _ts()
        self.last_retrain_ts = ts
        self.retrain_history.append({
            "ts": ts, "reason": reason, "runs_at_retrain": self.total_runs})
        if len(self.retrain_history) > 100:
            self.retrain_history = self.retrain_history[-100:]
        log.info(f"[Retrain] Marked retrained at {ts}. Reason: {reason[:80]}")

    def update_config(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in DEFAULTS:
                self.config[k] = v
                log.info(f"[Retrain] Config updated: {k}={v}")

    def save(self) -> None:
        _save_json(_RUN_LEDGER_PATH, self.ledger.to_dict())
        _save_json(_RETRAIN_META_PATH, {
            "version":         self.VERSION,
            "total_runs":      self.total_runs,
            "last_retrain_ts": self.last_retrain_ts,
            "config":          self.config,
            "retrain_history": self.retrain_history,
            "ledger_size":     len(self.ledger),
            "last_saved":      _ts(),
        })
        log.info(f"[Retrain] Saved — runs={self.total_runs}, ledger={len(self.ledger)}")

    def full_report(self) -> Dict:
        return {
            "version":         self.VERSION,
            "total_runs":      self.total_runs,
            "ledger_size":     len(self.ledger),
            "last_retrain_ts": self.last_retrain_ts,
            "retrain_count":   len(self.retrain_history),
            "config":          self.config,
            # v2.0
            "scorer_ready":    _global_retrain_scorer._clf is not None,
            "learned_thresholds": _global_threshold_learner._learned,
        }


# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as fh: return json.load(fh)
    except Exception as exc:
        log.warning(f"[Retrain] Could not load {path}: {exc}")
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
    except Exception as exc:
        log.warning(f"[Retrain] Could not save {path}: {exc}")


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    raise TypeError(f"Not serialisable: {type(obj)}")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_result(cv_mean=0.85, cv_std=0.03, confidence_label="High",
                      overfitting=False, leakage=False, n_rows=5000,
                      n_cols=22, n_classes=2, missing_pct=4.0,
                      best_model="LightGBM") -> Dict:
    return {
        "problem_type": "classification", "best_model_name": best_model,
        "stacking_model": "N/A", "scale_tier": 1,
        "dataset_diagnostics": {"n_rows": n_rows, "n_cols": n_cols,
                                 "overall_missing_pct": missing_pct, "n_classes": n_classes},
        "performance": {
            "cv_score_mean": cv_mean, "cv_score_std": cv_std,
            "confidence_score": 0.80, "confidence_label": confidence_label,
            "overfitting": overfitting, "leakage_detected": leakage, "n_cv_folds": 3,
            "feature_selection": {"original_features": n_cols, "final_features": n_cols},
        },
        "baseline_alert": {"triggered": False},
    }


def _run_tests() -> None:
    import tempfile
    print("\n── retrain_model.py v2.0 self-tests ──")

    # ── 1. _extract_snapshot ─────────────────────────────────────────────────
    snap = _extract_snapshot(_make_mock_result())
    assert snap["cv_mean"] == 0.85
    assert snap["n_rows"]  == 5000
    print(f"✓ _extract_snapshot OK")

    # ── 2. CATASTROPHIC gate — absolute CV floor ──────────────────────────────
    snap_cat = {"cv_mean": 0.25, "leakage_detected": False}
    alerts   = [Alert("PERFORMANCE_DROP", SEVERITY_HIGH, False, "OK")]
    should, reason, sev = _aggregate_alerts(alerts, snapshot=snap_cat)
    assert should
    assert "CATASTROPHIC" in reason
    print(f"✓ Catastrophic CV floor gate fires correctly")

    # ── 3. Leakage catastrophic gate ─────────────────────────────────────────
    snap_leak = {"cv_mean": 0.90, "leakage_detected": True}
    should2, _, _ = _aggregate_alerts([], snapshot=snap_leak)
    assert should2
    print(f"✓ Catastrophic leakage gate fires correctly")

    # ── 4. CRITICAL trigger fires unconditionally ─────────────────────────────
    a_crit = Alert("LEAKAGE_DETECTED", SEVERITY_CRITICAL, True, "Leakage!")
    should3, _, sev3 = _aggregate_alerts([a_crit])
    assert should3 and sev3 == SEVERITY_CRITICAL
    print(f"✓ CRITICAL trigger unconditional OK")

    # ── 5. RetrainScorer returns None before training ─────────────────────────
    rs  = RetrainScorer()
    res = rs.predict([Alert("HIGH_VARIANCE", SEVERITY_MEDIUM, True, "High var")])
    assert res is None
    print(f"✓ RetrainScorer returns None before training")

    # ── 6. Fallback count logic when no scorer ────────────────────────────────
    medium1 = Alert("HIGH_VARIANCE",   SEVERITY_MEDIUM, True, "Var high")
    medium2 = Alert("OVERFIT_STREAK",  SEVERITY_MEDIUM, True, "Overfit")
    should4, reason4, sev4 = _aggregate_alerts([medium1, medium2])
    assert should4 and sev4 == SEVERITY_MEDIUM
    print(f"✓ Fallback MEDIUM×2 → retrain=True")

    only1 = Alert("HIGH_VARIANCE", SEVERITY_MEDIUM, True, "One medium")
    should5, _, _ = _aggregate_alerts([only1])
    assert not should5
    print(f"✓ Fallback MEDIUM×1 → retrain=False")

    # ── 7. AdaptiveThresholdLearner defaults ─────────────────────────────────
    atl = AdaptiveThresholdLearner()
    assert atl.get("drop_threshold",     0.05) == 0.05
    assert atl.get("variance_threshold", 0.08) == 0.08
    print(f"✓ AdaptiveThresholdLearner defaults OK")

    # ── 8. RetrainController full lifecycle ───────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        global _RUN_LEDGER_PATH, _RETRAIN_META_PATH, _RETRAIN_STATE_DIR
        _orig = (_RUN_LEDGER_PATH, _RETRAIN_META_PATH, _RETRAIN_STATE_DIR)
        _RETRAIN_STATE_DIR = Path(tmp)
        _RUN_LEDGER_PATH   = _RETRAIN_STATE_DIR / "run_ledger.json"
        _RETRAIN_META_PATH = _RETRAIN_STATE_DIR / "retrain_meta.json"

        rc = RetrainController.load_or_create()
        assert rc.total_runs == 0

        # Prime with stable runs
        for _ in range(10):
            rc.record(_make_mock_result(cv_mean=0.90))

        # Catastrophic drop
        drop_result = _make_mock_result(cv_mean=0.25)
        should, reason = rc.should_retrain(drop_result)
        assert should, f"Expected retrain, got False. Reason: {reason}"
        print(f"✓ RetrainController catastrophic drop → should_retrain=True")

        rc.mark_retrained(reason="catastrophic_cv_drop")
        assert rc.last_retrain_ts is not None

        rc.save()
        rc2 = RetrainController.load_or_create()
        assert rc2.total_runs == 10
        print(f"✓ RetrainController save/load OK  runs={rc2.total_runs}")

        # full_report v2.0 fields
        report = rc2.full_report()
        assert "scorer_ready"         in report
        assert "learned_thresholds"   in report
        print(f"✓ full_report v2.0 fields OK")

        _RUN_LEDGER_PATH, _RETRAIN_META_PATH, _RETRAIN_STATE_DIR = _orig

    # ── 9. TIME_ELAPSED ───────────────────────────────────────────────────────
    ledger5 = RunLedger()
    old_ts  = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    snap_t  = _extract_snapshot(_make_mock_result())
    alert_t = _detect_time_elapsed(snap_t, ledger5, max_days_without_retrain=30,
                                   last_retrain_ts=old_ts)
    assert alert_t.fired
    print(f"✓ TIME_ELAPSED fires (elapsed={alert_t.values['elapsed_days']}d)")

    print("\n✓ All retrain_model.py v2.0 self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()