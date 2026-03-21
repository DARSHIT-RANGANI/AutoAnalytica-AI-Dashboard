"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  rl_agent.py  — v2.0  (AutoAnalytica v5.5)                                 ║
║                                                                              ║
║  Persistent Cross-Run Reinforcement Learning Agent                           ║
║                                                                              ║
║  Design                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  The agent in automl_service.py (Multi-Armed Bandit, epsilon-greedy)        ║
║  operates WITHIN a single run — it selects among model candidates and       ║
║  forgets everything when Python exits.                                       ║
║                                                                              ║
║  This module operates ACROSS runs — it learns which pipeline-level          ║
║  actions produce the best outcomes over many datasets and sessions,          ║
║  and survives restarts via JSON state in rl_state/.                         ║
║                                                                              ║
║  Layered architecture                                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Layer 0 (automl_service.py) : per-run MAB — model arm selection            ║
║  Layer 1 (rl_agent.py)       : cross-run Q-table — pipeline action          ║
║                                 (feature engineering, CV strategy, etc.)    ║
║                                                                              ║
║  Action space (pipeline-level decisions)                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BOOST_ENSEMBLE        → increase stacking top_n to 5                       ║
║  INCREASE_CV_FOLDS     → use 5-fold instead of 3-fold                       ║
║  APPLY_DISTILLATION    → force distillation even below size threshold       ║
║  AGGRESSIVE_FS         → tighten feature selection (lower k)                ║
║  CONSERVATIVE_FS       → loosen feature selection (higher k)                ║
║  PRIORITISE_BOOSTING   → push XGB/LGBM/CB to front of candidate order      ║
║  PRIORITISE_LINEAR     → push linear models to front                        ║
║  NO_OVERRIDE           → accept default pipeline (null action)              ║
║                                                                              ║
║  State bucketing                                                             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Continuous state dims (rows, features, imbalance, etc.) are bucketed       ║
║  into discrete bins so the Q-table stays finite. Each (bucket, action)      ║
║  pair has its own avg_reward tracked via Welford incremental mean.          ║
║                                                                              ║
║  v2.0 Changes                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  + ColdStartPredictor   : GradientBoosting replaces if-else heuristic       ║
║  + PenaltyWeightModel   : Ridge regression replaces hardcoded penalties      ║
║  + QTable variance      : Welford variance tracked per entry for UCB        ║
║  + Adaptive UCB c param : scales with per-bucket reward variance             ║
║  + Adaptive near-best   : band computed from bucket signal spread            ║
║                                                                              ║
║  Persistence                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  State dir : rl_state/                                                       ║
║  Files     : rl_state/q_table.json   — Q-table with Welford stats           ║
║              rl_state/agent_meta.json — run counter, epsilon, history       ║
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
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_RL_STATE_DIR    = Path("rl_state")
_Q_TABLE_PATH    = _RL_STATE_DIR / "q_table.json"
_AGENT_META_PATH = _RL_STATE_DIR / "agent_meta.json"

_RL_STATE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS: List[str] = [
    "NO_OVERRIDE",
    "BOOST_ENSEMBLE",
    "INCREASE_CV_FOLDS",
    "APPLY_DISTILLATION",
    "AGGRESSIVE_FS",
    "CONSERVATIVE_FS",
    "PRIORITISE_BOOSTING",
    "PRIORITISE_LINEAR",
]

# Human-readable descriptions (used in logs and meta_insight output)
ACTION_DESCRIPTIONS: Dict[str, str] = {
    "NO_OVERRIDE":         "Accept default pipeline — no action.",
    "BOOST_ENSEMBLE":      "Increase stacking top_n → 5 base models.",
    "INCREASE_CV_FOLDS":   "Use 5-fold CV for more stable estimates.",
    "APPLY_DISTILLATION":  "Force knowledge distillation regardless of size.",
    "AGGRESSIVE_FS":       "Tighten feature selection; fewer, higher-signal features.",
    "CONSERVATIVE_FS":     "Loosen feature selection; retain more features.",
    "PRIORITISE_BOOSTING": "Push XGBoost/LightGBM/CatBoost to front of candidate list.",
    "PRIORITISE_LINEAR":   "Push linear models to front of candidate list.",
}

# ─────────────────────────────────────────────────────────────────────────────
# STATE BUCKETING
# ─────────────────────────────────────────────────────────────────────────────
# Continuous dimensions → discrete bucket labels
# The Cartesian product of all bucket labels forms the state key.

_ROW_BINS: List[Tuple[int, str]] = [
    (200,      "tiny"),        # < 200
    (10_000,   "small"),       # 200 – 10k
    (100_000,  "medium"),      # 10k – 100k
    (math.inf, "large"),       # > 100k
]

_FEAT_BINS: List[Tuple[int, str]] = [
    (10,       "low"),
    (50,       "medium"),
    (200,      "high"),
    (math.inf, "very_high"),
]

_IMBALANCE_BINS: List[Tuple[float, str]] = [
    (0.60,     "balanced"),    # majority ≤ 60 %
    (0.80,     "moderate"),    # 60–80 %
    (math.inf, "imbalanced"),  # > 80 %
]

_MISSING_BINS: List[Tuple[float, str]] = [
    (0.05,     "clean"),       # ≤ 5 % missing
    (0.20,     "moderate"),
    (math.inf, "heavy"),
]


def _bucket(value: float,
            bins: List[Tuple[float, str]],
            default: str = "unknown") -> str:
    """Map a continuous value to the label of the first bin it falls below."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    for threshold, label in bins:
        if value < threshold:
            return label
    return bins[-1][1]


def state_to_bucket(state: Dict[str, Any]) -> str:
    """
    Convert a raw state dict (from _build_rl_state or _build_rl_state_from_result)
    into a hashable bucket string for Q-table lookup.

    Input keys (all optional with safe defaults):
        rows, features, missing_ratio, class_imbalance, problem_type, tier
    """
    rows      = float(state.get("rows") or state.get("n_rows", 0))
    features  = float(state.get("features", 0))
    missing   = float(state.get("missing_ratio", 0.0))
    imbalance = state.get("class_imbalance")
    prob_type = state.get("problem_type", "unknown")
    tier      = int(state.get("tier", 1))

    row_b  = _bucket(rows,     _ROW_BINS)
    feat_b = _bucket(features, _FEAT_BINS)
    mis_b  = _bucket(missing,  _MISSING_BINS)

    if imbalance is not None:
        imb_b = _bucket(float(imbalance), _IMBALANCE_BINS)
    else:
        imb_b = "n/a"   # regression — no class imbalance

    bucket = f"{prob_type}|t{tier}|rows:{row_b}|feat:{feat_b}|mis:{mis_b}|imb:{imb_b}"
    return bucket


# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE  (Welford incremental mean + variance per (bucket, action) pair)
# ─────────────────────────────────────────────────────────────────────────────

class QTable:
    """
    Finite Q-table mapping (state_bucket, action) → {trials, avg_reward, m2}.

    v2.0: Extends Welford tracking to also capture M2 (sum of squared
    deviations) so callers can compute per-entry reward variance.
    Variance feeds the adaptive UCB exploration bonus in choose_action().

    Serialised as:
        {
          "state_bucket_1": {
            "ACTION_A": {"trials": 5, "avg_reward": 0.83, "m2": 0.012},
            ...
          },
          ...
        }
    """

    def __init__(self, data: Dict[str, Dict[str, Dict]] | None = None) -> None:
        # {bucket: {action: {trials, avg_reward, m2}}}
        self._table: Dict[str, Dict[str, Dict]] = data or {}

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, bucket: str, action: str) -> Dict:
        return self._table.get(bucket, {}).get(
            action, {"trials": 0, "avg_reward": 0.0, "m2": 0.0}
        )

    def avg_reward(self, bucket: str, action: str) -> float:
        return self.get(bucket, action)["avg_reward"]

    def trials(self, bucket: str, action: str) -> int:
        return self.get(bucket, action)["trials"]

    def variance(self, bucket: str, action: str) -> float:
        """
        Sample variance of observed rewards for this (bucket, action) pair.
        Returns 0.0 when trials < 2 (variance undefined for single sample).
        """
        entry = self.get(bucket, action)
        n = entry["trials"]
        if n < 2:
            return 0.0
        return entry.get("m2", 0.0) / (n - 1)

    def all_actions_for(self, bucket: str) -> Dict[str, Dict]:
        return self._table.get(bucket, {})

    # ── Write (Welford update — mean + M2) ───────────────────────────────────

    def update(self, bucket: str, action: str, reward: float) -> None:
        """
        Welford online algorithm extended to track M2 alongside the mean.

        M2 accumulates the sum of squared deviations from the running mean,
        enabling numerically stable variance computation without storing the
        full reward history.
        """
        if bucket not in self._table:
            self._table[bucket] = {}
        if action not in self._table[bucket]:
            self._table[bucket][action] = {"trials": 0, "avg_reward": 0.0, "m2": 0.0}

        entry  = self._table[bucket][action]
        entry["trials"] += 1
        n      = entry["trials"]
        delta  = reward - entry["avg_reward"]
        entry["avg_reward"] += delta / n
        delta2 = reward - entry["avg_reward"]       # updated mean
        entry["m2"] = entry.get("m2", 0.0) + delta * delta2

        log.debug(
            f"[QTable] ({bucket}, {action}) → "
            f"trials={n}, avg={entry['avg_reward']:.4f}, "
            f"var={self.variance(bucket, action):.5f}"
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return self._table

    @classmethod
    def from_dict(cls, data: Dict) -> "QTable":
        return cls(data)

    def n_states(self) -> int:
        return len(self._table)

    def n_entries(self) -> int:
        return sum(len(v) for v in self._table.values())


# ─────────────────────────────────────────────────────────────────────────────
# UCB1 HELPER  (mirrors automl_service._ucb1_score)
# ─────────────────────────────────────────────────────────────────────────────

def _ucb1(avg_reward: float, trials: int,
          total_trials: int, c: float = 1.41) -> float:
    if trials == 0:
        return float("inf")
    return avg_reward + c * math.sqrt(math.log(max(total_trials, 1)) / trials)


# ─────────────────────────────────────────────────────────────────────────────
# COLD-START PREDICTOR  (model-based replacement for rule-based heuristic)
# ─────────────────────────────────────────────────────────────────────────────

class ColdStartPredictor:
    """
    GradientBoosting classifier that learns which pipeline action produces
    the best reward for unseen state buckets — trained lazily from run_history.

    Replaces the hard if-else chain in _cold_start_heuristic() once
    MIN_SAMPLES training examples are available.  Falls back gracefully
    to the rule heuristic when the model is absent or under-confident.

    Training signal
    ───────────────
    For each state bucket in run_history, the action with the highest
    average reward is taken as the label.  The bucket string is decoded
    back to a 6-dim numeric vector using midpoint approximations for each
    bin — enough signal for a shallow tree ensemble.
    """

    MIN_SAMPLES      = 20    # safety gate — must see N runs before trusting model
    RETRAIN_EVERY    = 10    # re-fit after every N new history entries
    CONFIDENCE_FLOOR = 0.30  # safety: ignore predictions below this probability

    # Mid-point proxies for decoding bucket strings → numeric features
    _ROW_MID  = {"tiny": 100, "small": 2_000, "medium": 30_000, "large": 200_000}
    _FEAT_MID = {"low": 5, "medium": 25, "high": 100, "very_high": 300}
    _MIS_MID  = {"clean": 0.01, "moderate": 0.12, "heavy": 0.40}
    _IMB_MID  = {"balanced": 0.55, "moderate": 0.70, "imbalanced": 0.90, "n/a": 0.50}

    def __init__(self) -> None:
        self._clf          = None          # sklearn model — set after first fit
        self._trained_on   = 0             # len(run_history) at last fit
        self._next_retrain = self.MIN_SAMPLES

    # ── Public ────────────────────────────────────────────────────────────────

    def maybe_fit(self, run_history: List[dict]) -> None:
        """Re-fit if enough new data has arrived since last training session."""
        if len(run_history) >= self._next_retrain:
            self._fit(run_history)
            self._next_retrain = len(run_history) + self.RETRAIN_EVERY

    def predict(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Return the predicted best action, or None if the model is absent
        or prediction confidence falls below CONFIDENCE_FLOOR (so the
        caller can fall back to the rule heuristic safely).
        """
        if self._clf is None:
            return None
        try:
            proba  = self._clf.predict_proba([self._state_to_vec(state)])[0]
            best_p = float(proba.max())
            if best_p < self.CONFIDENCE_FLOOR:
                log.debug(
                    f"[ColdStartPredictor] Low confidence ({best_p:.3f}) "
                    f"— deferring to rule heuristic."
                )
                return None
            action = str(self._clf.classes_[int(proba.argmax())])
            if action not in ACTIONS:
                return None
            log.info(f"[ColdStartPredictor] → {action}  (p={best_p:.3f})")
            return action
        except Exception as exc:
            log.warning(f"[ColdStartPredictor] predict failed: {exc}")
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit(self, run_history: List[dict]) -> None:
        from collections import defaultdict
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            log.warning("[ColdStartPredictor] sklearn not available — skipping fit.")
            return

        # For each bucket, label = action with highest average reward
        bucket_actions: Dict[str, Dict[str, List[float]]] = \
            defaultdict(lambda: defaultdict(list))
        for entry in run_history:
            b = entry.get("bucket")
            a = entry.get("action")
            r = entry.get("reward")
            if b and a and r is not None:
                bucket_actions[b][a].append(float(r))

        X: List[List[float]] = []
        y: List[str]          = []
        for bucket, actions in bucket_actions.items():
            best_a = max(actions, key=lambda a: sum(actions[a]) / len(actions[a]))
            fv = self._bucket_to_vec(bucket)
            if fv is not None:
                X.append(fv)
                y.append(best_a)

        if len(X) < 5 or len(set(y)) < 2:
            log.debug("[ColdStartPredictor] Insufficient label variety — skipping fit.")
            return

        clf = GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )
        clf.fit(X, y)
        self._clf        = clf
        self._trained_on = len(run_history)
        log.info(
            f"[ColdStartPredictor] Fitted — "
            f"{len(X)} buckets · {len(run_history)} history entries · "
            f"{len(set(y))} distinct actions seen."
        )

    def _state_to_vec(self, state: Dict[str, Any]) -> List[float]:
        rows      = float(state.get("rows") or state.get("n_rows") or 1_000)
        features  = float(state.get("features") or 10)
        missing   = float(state.get("missing_ratio") or 0.0)
        imbalance = float(state.get("class_imbalance") or 0.5)
        prob_type = 1.0 if state.get("problem_type") == "regression" else 0.0
        tier      = float(state.get("tier", 1)) / 4.0
        return [
            math.log10(max(rows,     1) + 1) / 7.0,
            math.log10(max(features, 1) + 1) / 4.0,
            min(missing,   1.0),
            min(imbalance, 1.0),
            prob_type,
            min(tier, 1.0),
        ]

    def _bucket_to_vec(self, bucket: str) -> Optional[List[float]]:
        """Reconstruct approximate numeric features from a bucket string."""
        try:
            parts: Dict[str, str] = {}
            for seg in bucket.split("|"):
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    parts[k] = v
            tier_segs = [s for s in bucket.split("|")
                         if s.startswith("t") and len(s) <= 3]
            tier      = float(tier_segs[0][1:]) if tier_segs else 1.0
            prob_type = 1.0 if bucket.startswith("regression") else 0.0
            rows_r  = math.log10(
                self._ROW_MID.get(parts.get("rows",  "small"),  2_000) + 1) / 7.0
            feats_r = math.log10(
                self._FEAT_MID.get(parts.get("feat", "medium"), 25)    + 1) / 4.0
            mis_r   = self._MIS_MID.get(parts.get("mis", "clean"),  0.01)
            imb_r   = self._IMB_MID.get(parts.get("imb", "n/a"),    0.50)
            return [rows_r, feats_r, mis_r, imb_r, prob_type, tier / 4.0]
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PENALTY WEIGHT MODEL  (replaces hardcoded penalties in build_reward)
# ─────────────────────────────────────────────────────────────────────────────

class PenaltyWeightModel:
    """
    Ridge regression that learns data-driven penalty weights for the three
    quality flags: overfitting, leakage, baseline_alert.

    Training signal
    ───────────────
    Each run_history entry that carries both a raw `confidence` score AND
    `penalty_components` flags provides one training sample:

        observed_penalty  =  confidence  −  reward

    Ridge regresses the three binary flags against the observed penalty:
        [overfit, leakage, baseline] → observed_penalty

    Learned coefficients → replace the hardcoded 0.20 / 0.15 / 0.10 in
    build_reward() so penalties adapt to what the data actually shows.

    Falls back to default values when fewer than MIN_SAMPLES labelled
    entries exist (safety: never leaves the system penalty-free).
    """

    MIN_SAMPLES = 30

    # Hard-coded fallback — active only before the model has enough data
    _DEFAULTS: Dict[str, float] = {
        "overfitting":    0.20,
        "leakage":        0.15,
        "baseline_alert": 0.10,
    }

    def __init__(self) -> None:
        self._weights: Dict[str, float] = dict(self._DEFAULTS)

    # ── Public ────────────────────────────────────────────────────────────────

    def maybe_fit(self, run_history: List[dict]) -> None:
        """Fit on history entries that contain confidence + penalty_components."""
        labelled = [
            e for e in run_history
            if e.get("confidence") is not None and e.get("penalty_components")
        ]
        if len(labelled) >= self.MIN_SAMPLES:
            self._fit(labelled)

    def penalty(self,
                overfitting:    bool,
                leakage:        bool,
                baseline_alert: bool) -> float:
        """Return the model-derived total penalty (always ≥ 0)."""
        total = (
            self._weights["overfitting"]    * float(overfitting)    +
            self._weights["leakage"]        * float(leakage)        +
            self._weights["baseline_alert"] * float(baseline_alert)
        )
        return max(0.0, total)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit(self, labelled: List[dict]) -> None:
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            log.warning("[PenaltyWeightModel] sklearn unavailable — keeping defaults.")
            return

        X, y = [], []
        for e in labelled:
            pc = e["penalty_components"]
            X.append([
                float(bool(pc.get("overfitting",    False))),
                float(bool(pc.get("leakage",        False))),
                float(bool(pc.get("baseline_alert", False))),
            ])
            # observed penalty = gap between raw confidence and final reward
            obs = max(0.0, float(e["confidence"]) - float(e["reward"]))
            y.append(obs)

        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y)

        keys = ["overfitting", "leakage", "baseline_alert"]
        # Clamp to [0, 0.50] — safety: prevent runaway or negative weights
        self._weights = {
            k: float(max(0.0, min(0.50, c)))
            for k, c in zip(keys, reg.coef_)
        }
        log.info(
            f"[PenaltyWeightModel] Fitted — weights={self._weights} "
            f"({len(labelled)} labelled samples)"
        )


# Global instance — shared by build_reward() and RLAgent so learned
# weights are reflected immediately in module-level calls.
_global_penalty_model: PenaltyWeightModel = PenaltyWeightModel()


# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Cross-run pipeline-level RL agent.

    Lifecycle
    ─────────
    agent = RLAgent.load_or_create()   # load from disk or fresh init
    action = agent.choose_action(state)
    # … run pipeline …
    agent.update(state, action, reward=confidence_score)
    agent.save()

    Epsilon schedule
    ────────────────
    Starts at eps_start (0.40), decays multiplicatively by eps_decay
    each time choose_action() is called, floored at eps_min (0.05).
    Persisted across sessions — the agent gets progressively less
    exploratory as it accumulates experience.

    v2.0 Adaptive UCB
    ─────────────────
    The UCB exploration bonus scales with the per-bucket reward variance:
        c = 1.41 × (1 + √variance)
    High-variance buckets receive a larger bonus — exploring uncertain
    regions more aggressively until the Q-estimates stabilise.

    v2.0 Adaptive near-best band
    ─────────────────────────────
    The near-best tiebreaker window is proportional to the reward spread
    within the current bucket instead of a fixed 0.005 gap.  This avoids
    over-collapsing to a single action in buckets with tight reward clusters.
    """

    VERSION = "2.0"

    def __init__(
        self,
        q_table:      QTable            | None = None,
        eps:          float                    = 0.40,
        eps_decay:    float                    = 0.995,
        eps_min:      float                    = 0.05,
        total_runs:   int                      = 0,
        run_history:  List[dict]         | None = None,
    ) -> None:
        self.q_table     = q_table or QTable()
        self.eps         = eps
        self.eps_decay   = eps_decay
        self.eps_min     = eps_min
        self.total_runs  = total_runs
        self.run_history: List[dict] = run_history or []
        self._last_state:  Optional[str] = None
        self._last_action: Optional[str] = None

        # v2.0 — model-based sub-components
        self._cold_start_model  = ColdStartPredictor()
        self._penalty_model     = _global_penalty_model

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls) -> "RLAgent":
        """Load from rl_state/ if files exist, otherwise create fresh."""
        q_table_data = _load_json_file(_Q_TABLE_PATH)
        agent_meta   = _load_json_file(_AGENT_META_PATH)

        if q_table_data or agent_meta:
            q_table = QTable.from_dict(q_table_data)
            agent = cls(
                q_table     = q_table,
                eps         = float(agent_meta.get("eps",        0.40)),
                eps_decay   = float(agent_meta.get("eps_decay",  0.995)),
                eps_min     = float(agent_meta.get("eps_min",    0.05)),
                total_runs  = int(agent_meta.get("total_runs",   0)),
                run_history = agent_meta.get("run_history",      []),
            )
            # v2.0 — warm up learned models from persisted history
            agent._cold_start_model.maybe_fit(agent.run_history)
            agent._penalty_model.maybe_fit(agent.run_history)

            log.info(
                f"[RLAgent] Loaded — runs={agent.total_runs}, "
                f"eps={agent.eps:.4f}, "
                f"q_states={q_table.n_states()}, "
                f"q_entries={q_table.n_entries()}"
            )
        else:
            agent = cls()
            log.info("[RLAgent] Fresh agent initialised.")
        return agent

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: Dict[str, Any]) -> str:
        """
        Select a pipeline-level action for the given state.

        Algorithm
        ─────────
        1. Bucket the state.
        2. If no entries exist for this bucket → model-based cold-start.
        3. ε-greedy with adaptive UCB1 tiebreaker.
             UCB c = 1.41 × (1 + √bucket_variance)  [v2.0]
        4. Near-best band = 5% of bucket reward spread  [v2.0]
        5. Decay epsilon.
        6. Record (state, action) for the upcoming update() call.
        """
        bucket  = state_to_bucket(state)
        known   = self.q_table.all_actions_for(bucket)
        untried = [a for a in ACTIONS if a not in known]

        # ── Cold-start: no data for this bucket ───────────────────────────────
        if not known:
            action = self._cold_start_heuristic(state)
            log.info(f"[RLAgent] COLD-START bucket={bucket!r}  → {action}")
            self._record_selection(bucket, action)
            return action

        # ── Force-explore untried actions (coverage guarantee) ────────────────
        if untried:
            action = untried[0]
            log.info(f"[RLAgent] FORCE-EXPLORE untried action → {action}")
            self._record_selection(bucket, action)
            self._decay_epsilon()
            return action

        # ── ε-greedy with adaptive UCB1 tiebreaker ────────────────────────────
        if random.random() < self.eps:
            action = random.choice(ACTIONS)
            log.info(f"[RLAgent] EXPLORE (eps={self.eps:.3f}) → {action}")
        else:
            total_t = sum(self.q_table.trials(bucket, a) for a in ACTIONS)

            # v2.0 — adaptive UCB exploration coefficient
            # Scales with the mean reward variance across known actions so
            # high-uncertainty buckets are explored more aggressively.
            mean_var = (
                sum(self.q_table.variance(bucket, a) for a in known) / len(known)
            )
            adaptive_c = 1.41 * (1.0 + math.sqrt(mean_var))

            # v2.0 — adaptive near-best band
            # Band = 5% of the reward spread in this bucket.
            # Prevents collapsing to one action in tight-cluster buckets.
            rewards    = [self.q_table.avg_reward(bucket, a) for a in known]
            spread     = max(rewards) - min(rewards)
            near_band  = max(0.005, 0.05 * spread)   # floor: 0.005 for safety

            best_r     = max(rewards)
            near_best  = [
                a for a in known
                if best_r - self.q_table.avg_reward(bucket, a) <= near_band
            ]

            if len(near_best) > 1:
                # Adaptive UCB1 tiebreaker among near-best
                action = max(
                    near_best,
                    key=lambda a: _ucb1(
                        self.q_table.avg_reward(bucket, a),
                        self.q_table.trials(bucket, a),
                        total_t,
                        c=adaptive_c,
                    ),
                )
                log.info(
                    f"[RLAgent] EXPLOIT+UCB1 → {action} "
                    f"(r={self.q_table.avg_reward(bucket, action):.4f}, "
                    f"c={adaptive_c:.3f}, band={near_band:.4f})"
                )
            else:
                action = max(known,
                             key=lambda a: self.q_table.avg_reward(bucket, a))
                log.info(
                    f"[RLAgent] EXPLOIT → {action} "
                    f"(r={self.q_table.avg_reward(bucket, action):.4f})"
                )

        self._record_selection(bucket, action)
        self._decay_epsilon()
        return action

    # ── Reward update ─────────────────────────────────────────────────────────

    def update(
        self,
        state:   Dict[str, Any],
        action:  str,
        reward:  float,
        *,
        clamp:   bool = True,
    ) -> None:
        """
        Back-propagate reward into the Q-table.

        reward  — typically the pipeline confidence_score ∈ [0, 1].
                  The caller may also pass a composite reward:
                    reward = confidence_score - 0.2 * int(overfitting)
                  This module does not compute reward internally —
                  the caller controls the signal.

        clamp=True (default) clips reward to [0, 1] to keep the Q-table
        bounded regardless of what the caller passes.

        v2.0 — triggers incremental model re-training every RETRAIN_EVERY runs.
        """
        if clamp:
            reward = max(0.0, min(1.0, float(reward)))

        bucket = state_to_bucket(state)
        self.q_table.update(bucket, action, reward)
        self.total_runs += 1

        entry = {
            "ts":      _ts(),
            "run":     self.total_runs,
            "bucket":  bucket,
            "action":  action,
            "reward":  round(reward, 4),
            "eps":     round(self.eps, 4),
        }
        self.run_history.append(entry)

        # Keep last 500 entries in memory
        if len(self.run_history) > 500:
            self.run_history = self.run_history[-500:]

        # v2.0 — re-fit learned sub-models incrementally
        self._cold_start_model.maybe_fit(self.run_history)
        self._penalty_model.maybe_fit(self.run_history)

        log.info(
            f"[RLAgent] Updated — run={self.total_runs}, "
            f"bucket={bucket!r}, action={action}, "
            f"reward={reward:.4f}, eps={self.eps:.4f}"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Flush Q-table and agent meta to rl_state/."""
        _save_json_file(_Q_TABLE_PATH, self.q_table.to_dict())
        meta = {
            "version":    self.VERSION,
            "eps":        round(self.eps, 6),
            "eps_decay":  self.eps_decay,
            "eps_min":    self.eps_min,
            "total_runs": self.total_runs,
            "run_history":self.run_history[-200:],   # persist last 200
            "q_stats": {
                "n_states":  self.q_table.n_states(),
                "n_entries": self.q_table.n_entries(),
            },
            "last_saved": _ts(),
        }
        _save_json_file(_AGENT_META_PATH, meta)
        log.info(
            f"[RLAgent] Saved — runs={self.total_runs}, "
            f"q_states={self.q_table.n_states()}, "
            f"eps={self.eps:.4f}"
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def top_actions(self, state: Dict[str, Any], n: int = 3) -> List[Dict]:
        """
        Return the top-n actions for a given state, ranked by avg_reward.
        Useful for meta_insight reporting.
        """
        bucket = state_to_bucket(state)
        known  = self.q_table.all_actions_for(bucket)
        if not known:
            return [{"action": a, "avg_reward": None, "trials": 0,
                     "description": ACTION_DESCRIPTIONS[a]}
                    for a in ACTIONS[:n]]
        ranked = sorted(known.items(),
                        key=lambda kv: kv[1]["avg_reward"], reverse=True)
        return [
            {
                "action":      act,
                "avg_reward":  round(stats["avg_reward"], 4),
                "trials":      stats["trials"],
                "description": ACTION_DESCRIPTIONS.get(act, ""),
            }
            for act, stats in ranked[:n]
        ]

    def full_report(self) -> Dict:
        """Full agent snapshot — suitable for the /agent_status API endpoint."""
        return {
            "version":    self.VERSION,
            "total_runs": self.total_runs,
            "epsilon":    round(self.eps, 4),
            "q_table": {
                "n_states":  self.q_table.n_states(),
                "n_entries": self.q_table.n_entries(),
            },
            "action_space": ACTIONS,
            "recent_history": self.run_history[-10:],
            # v2.0 — expose learned penalty weights in status report
            "learned_penalty_weights": self._penalty_model._weights,
            "cold_start_model_ready":  self._cold_start_model._clf is not None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cold_start_heuristic(self, state: Dict[str, Any]) -> str:
        """
        v2.0 — Model-first cold-start.

        Attempts ColdStartPredictor first; falls back to the minimal
        rule-based heuristic only when the model is absent, under-confident,
        or sklearn is unavailable.

        Rules retained here are a SAFETY NET, not primary logic.
        """
        # ── Primary: learned model ────────────────────────────────────────────
        model_action = self._cold_start_model.predict(state)
        if model_action is not None:
            return model_action

        # ── Fallback: minimal safety rules (kept for cold boot only) ──────────
        rows      = float(state.get("rows") or state.get("n_rows", 0))
        features  = float(state.get("features", 0))
        imbalance = state.get("class_imbalance")
        prob_type = state.get("problem_type", "unknown")

        # Safety net — only fires when the model has no prediction
        if prob_type == "classification" and imbalance and imbalance > 0.80:
            return "AGGRESSIVE_FS"
        if rows >= 50_000:
            return "BOOST_ENSEMBLE"
        if rows < 500:
            return "PRIORITISE_LINEAR"
        if features > rows:
            return "CONSERVATIVE_FS"
        if rows >= 10_000:
            return "PRIORITISE_BOOSTING"
        return "NO_OVERRIDE"

    def _decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def _record_selection(self, bucket: str, action: str) -> None:
        self._last_state  = bucket
        self._last_action = action


# ─────────────────────────────────────────────────────────────────────────────
# ACTION → CONTEXT OVERRIDE TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

def action_to_context_override(action: str) -> Dict[str, Any]:
    """
    Translate a pipeline action string into a context_overrides dict
    that run_full_pipeline() / run_automl_with_agents() understands.

    The receiving functions read these keys from context_overrides and
    adjust their behaviour accordingly.  Keys not listed here are ignored.

    This is the bridge between Layer 1 (RL agent) and Layer 0 (pipeline).
    """
    overrides: Dict[str, Any] = {}

    if action == "BOOST_ENSEMBLE":
        overrides["ensemble_top_n"] = 5          # _build_stacking_model top_n

    elif action == "INCREASE_CV_FOLDS":
        overrides["n_cv_folds"] = 5              # cross_validate n_folds

    elif action == "APPLY_DISTILLATION":
        overrides["force_distillation"] = True   # distill_model bypass threshold

    elif action == "AGGRESSIVE_FS":
        overrides["fs_selectk_max"] = 20         # auto_feature_selection cap k

    elif action == "CONSERVATIVE_FS":
        overrides["fs_selectk_max"] = 100        # auto_feature_selection raise k

    elif action == "PRIORITISE_BOOSTING":
        overrides["meta_priority_hint"] = "boosting"

    elif action == "PRIORITISE_LINEAR":
        overrides["meta_priority_hint"] = "linear"

    # NO_OVERRIDE → empty dict → pipeline uses all defaults
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# REWARD BUILDER  (helper for callers)
# ─────────────────────────────────────────────────────────────────────────────

def build_reward(
    confidence_score:  float,
    overfitting:       bool  = False,
    leakage_detected:  bool  = False,
    baseline_alert:    bool  = False,
    *,
    run_history:       Optional[List[dict]] = None,
) -> float:
    """
    Compose a scalar reward signal from pipeline outcome metrics.

    v2.0 — penalty weights are learned by PenaltyWeightModel (Ridge)
    instead of being hardcoded.  The model is triggered to re-fit whenever
    run_history is supplied and contains sufficient labelled entries.

    Falls back to default weights (0.20 / 0.15 / 0.10) automatically
    when fewer than PenaltyWeightModel.MIN_SAMPLES entries are available.

    Parameters
    ──────────
    confidence_score  — base score ∈ [0, 1]
    overfitting       — train/CV gap > 0.10
    leakage_detected  — |r| > 0.90 features found
    baseline_alert    — model barely beats majority class
    run_history       — optional; if supplied, triggers incremental re-fit

    Result clamped to [0, 1].
    """
    if run_history is not None:
        _global_penalty_model.maybe_fit(run_history)

    penalty = _global_penalty_model.penalty(
        overfitting    = overfitting,
        leakage        = leakage_detected,
        baseline_alert = baseline_alert,
    )
    reward = max(0.0, min(1.0, float(confidence_score) - penalty))

    log.debug(
        f"[Reward] base={confidence_score:.4f}  "
        f"overfit={overfitting}  leak={leakage_detected}  "
        f"baseline_alert={baseline_alert}  "
        f"weights={_global_penalty_model._weights}  → {reward:.4f}"
    )
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# JSON PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_file(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    except Exception as exc:
        log.warning(f"[RLAgent] Could not load {path}: {exc}")
    return {}


def _save_json_file(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
    except Exception as exc:
        log.warning(f"[RLAgent] Could not save {path}: {exc}")


def _json_default(obj: Any) -> Any:
    import math as _math
    if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
        return None
    raise TypeError(f"Not serialisable: {type(obj)}")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    print("\n── rl_agent.py self-tests (v2.0) ──")
    import tempfile, os

    # ── 1. State bucketing ────────────────────────────────────────────────────
    state_cls = {
        "rows": 500, "features": 25, "missing_ratio": 0.02,
        "class_imbalance": 0.65, "problem_type": "classification", "tier": 1,
    }
    bucket = state_to_bucket(state_cls)
    assert "classification" in bucket, f"Bad bucket: {bucket}"
    assert "rows:small"  in bucket, f"Bad row bucket: {bucket}"
    assert "feat:medium" in bucket, f"Bad feat bucket: {bucket}"
    print(f"✓ state_to_bucket OK  →  {bucket}")

    state_reg = {
        "rows": 1200, "features": 8, "missing_ratio": 0.30,
        "class_imbalance": None, "problem_type": "regression", "tier": 1,
    }
    bucket_r = state_to_bucket(state_reg)
    assert "regression" in bucket_r
    assert "imb:n/a"    in bucket_r
    print(f"✓ regression bucket OK  →  {bucket_r}")

    # ── 2. QTable Welford update + variance ───────────────────────────────────
    qt = QTable()
    for r in [0.8, 0.6, 0.9]:
        qt.update("test_bucket", "BOOST_ENSEMBLE", r)
    expected_avg = (0.8 + 0.6 + 0.9) / 3
    assert abs(qt.avg_reward("test_bucket", "BOOST_ENSEMBLE") - expected_avg) < 1e-9
    assert qt.trials("test_bucket", "BOOST_ENSEMBLE") == 3
    var = qt.variance("test_bucket", "BOOST_ENSEMBLE")
    assert var > 0, f"Expected positive variance, got {var}"
    print(f"✓ QTable Welford OK  avg={qt.avg_reward('test_bucket','BOOST_ENSEMBLE'):.4f}  var={var:.5f}")

    # ── 3. QTable serialisation round-trip ───────────────────────────────────
    d   = qt.to_dict()
    qt2 = QTable.from_dict(d)
    assert qt2.avg_reward("test_bucket", "BOOST_ENSEMBLE") == qt.avg_reward("test_bucket", "BOOST_ENSEMBLE")
    assert abs(qt2.variance("test_bucket", "BOOST_ENSEMBLE") - var) < 1e-9
    print("✓ QTable round-trip OK (incl. variance)")

    # ── 4. ColdStartPredictor fallback (no data) ─────────────────────────────
    csp = ColdStartPredictor()
    pred = csp.predict({"rows": 500, "problem_type": "classification",
                        "class_imbalance": 0.55, "features": 20})
    assert pred is None, "Expected None before training"
    print("✓ ColdStartPredictor returns None before training OK")

    # ── 5. RLAgent cold-start falls back to rules (no history) ───────────────
    agent = RLAgent()
    a = agent._cold_start_heuristic({"rows": 100, "problem_type": "classification", "class_imbalance": 0.55})
    assert a == "PRIORITISE_LINEAR", f"Expected PRIORITISE_LINEAR, got {a}"
    a = agent._cold_start_heuristic({"rows": 60_000, "problem_type": "regression"})
    assert a == "BOOST_ENSEMBLE", f"Expected BOOST_ENSEMBLE, got {a}"
    a = agent._cold_start_heuristic({"rows": 3000, "problem_type": "classification", "class_imbalance": 0.90})
    assert a == "AGGRESSIVE_FS", f"Expected AGGRESSIVE_FS, got {a}"
    print("✓ Cold-start fallback rules OK")

    # ── 6. PenaltyWeightModel — defaults before training ─────────────────────
    pwm = PenaltyWeightModel()
    p   = pwm.penalty(overfitting=True, leakage=True, baseline_alert=False)
    assert abs(p - 0.35) < 1e-9, f"Expected 0.35, got {p}"
    print(f"✓ PenaltyWeightModel defaults OK  (overfit+leak penalty={p:.2f})")

    # ── 7. build_reward uses global penalty model ─────────────────────────────
    r = build_reward(0.85, overfitting=True, leakage_detected=True)
    assert abs(r - 0.50) < 1e-9, f"Expected 0.50, got {r}"
    r2 = build_reward(0.95, overfitting=False)
    assert abs(r2 - 0.95) < 1e-9
    print(f"✓ build_reward OK  (0.85-0.20-0.15={r:.2f})")

    # ── 8. choose_action coverage guarantee ──────────────────────────────────
    agent2  = RLAgent()
    bucket2 = "classification|t1|rows:small|feat:medium|mis:clean|imb:balanced"
    for act in ACTIONS[:-1]:
        agent2.q_table.update(bucket2, act, 0.7)
    untried = [a for a in ACTIONS if a not in agent2.q_table.all_actions_for(bucket2)]
    assert len(untried) == 1 and untried[0] == ACTIONS[-1]
    print(f"✓ Coverage guarantee OK — untried: {untried}")

    # ── 9. Adaptive UCB — c scales with variance ──────────────────────────────
    agent3 = RLAgent()
    bkt3   = "classification|t1|rows:medium|feat:medium|mis:clean|imb:balanced"
    # inject high-variance rewards for one action
    for r_val in [0.1, 0.9, 0.5, 0.8, 0.2]:
        agent3.q_table.update(bkt3, "BOOST_ENSEMBLE", r_val)
    # inject low-variance rewards for another
    for r_val in [0.7, 0.71, 0.69, 0.70, 0.705]:
        agent3.q_table.update(bkt3, "NO_OVERRIDE", r_val)
    var_high = agent3.q_table.variance(bkt3, "BOOST_ENSEMBLE")
    var_low  = agent3.q_table.variance(bkt3, "NO_OVERRIDE")
    assert var_high > var_low, "High-variance action should have larger variance"
    print(f"✓ Adaptive UCB variance tracking OK  (high={var_high:.4f} > low={var_low:.6f})")

    # ── 10. action_to_context_override ───────────────────────────────────────
    ctx = action_to_context_override("BOOST_ENSEMBLE")
    assert ctx == {"ensemble_top_n": 5}
    ctx2 = action_to_context_override("NO_OVERRIDE")
    assert ctx2 == {}
    print("✓ action_to_context_override OK")

    # ── 11. Save / load round-trip (temp dir) ────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        global _Q_TABLE_PATH, _AGENT_META_PATH, _RL_STATE_DIR
        _orig_q   = _Q_TABLE_PATH
        _orig_m   = _AGENT_META_PATH
        _orig_dir = _RL_STATE_DIR

        _RL_STATE_DIR    = Path(tmp)
        _Q_TABLE_PATH    = _RL_STATE_DIR / "q_table.json"
        _AGENT_META_PATH = _RL_STATE_DIR / "agent_meta.json"

        a1 = RLAgent()
        a1.q_table.update("bucket_x", "NO_OVERRIDE", 0.77)
        a1.total_runs = 7
        a1.save()

        a2 = RLAgent.load_or_create()
        assert a2.total_runs == 7
        assert abs(a2.q_table.avg_reward("bucket_x", "NO_OVERRIDE") - 0.77) < 1e-9

        _Q_TABLE_PATH    = _orig_q
        _AGENT_META_PATH = _orig_m
        _RL_STATE_DIR    = _orig_dir

    print("✓ Save/load round-trip OK")
    print("\n✓ All rl_agent.py v2.0 self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()


# ══════════════════════════════════════════════════════════════════════════════
# STATE VECTOR  (v5.5 — consumed by feature_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════
#
#  build_state_vector() converts a raw automl_service result dict into a
#  fixed-length numeric array that can be fed directly to any sklearn or
#  numpy-based model.
#
#  Layout (STATE_DIM = 20 dims, all values ∈ [0, 1]):
#  ─────────────────────────────────────────────────
#  [ 0] n_rows_norm       log10(n_rows+1) / 7.0
#  [ 1] n_feats_norm      log10(n_feats+1) / 4.0
#  [ 2] missing_ratio     overall_missing_pct / 100
#  [ 3] imbalance         class_imbalance (0.5 for regression)
#  [ 4] prob_type         0=classification, 1=regression
#  [ 5] tier_norm         scale_tier / 4.0
#  [ 6] small_dataset     1 if n_rows < 200 else 0
#  [ 7] cv_score          cv_score_mean
#  [ 8] cv_std_n          cv_score_std / 0.5  (clamped)
#  [ 9] test_score        accuracy (cls) or R2 (reg)
#  [10] overfit_gap_n     max(0, train_score - cv_score) / 0.4  (clamped)
#  [11] confidence        confidence_score
#  [12] leakage           1 if leakage_detected else 0
#  [13] baseline_gap_n    0.5 + signed_gap  (shifted so 0.5 = at baseline)
#  [14] feat_reduction    1 - (final_features / original_features)
#  [15] model_family_n    _model_family_index(best_model) / 4.0
#  [16] stacking          1 if "Stacking" in best_model else 0
#  [17] fs_steps_n        len(steps_applied) / 5.0
#  [18] n_iters_n         len(agent_history) / 30.0  (clamped)
#  [19] avg_eff_score     mean(effective_score) across RL iterations
# ══════════════════════════════════════════════════════════════════════════════

import numpy as _np

STATE_DIM: int = 20   # exported constant — feature_extractor.py reads this

# Model-family index map (same as feature_extractor._model_family_index)
_FAMILY_KEYWORDS: dict = {
    0: ["logistic", "linear", "ridge", "lasso", "elastic", "bayesian",
        "sgd", "linearsvc", "linearsvr"],
    1: ["randomforest", "extratrees", "decisiontree", "gradientboosting",
        "histgradient", "adaboost", "bagging"],
    2: ["xgb", "lgbm", "lightgbm", "catboost"],
    3: ["gaussiannb", "bernoullinb", "multinomialnb", "qda"],
}


def _family_index(model_name: str) -> int:
    name_lc = model_name.lower().replace("_", "").replace(" ", "")
    for idx, kws in _FAMILY_KEYWORDS.items():
        if any(kw in name_lc for kw in kws):
            return idx
    return 4   # stacking / unknown


def _sv(v, default: float = 0.0) -> float:
    """Safe float with NaN/Inf guard."""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _sc(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi]."""
    return float(max(lo, min(hi, v)))


def _ln(raw: float, scale: float) -> float:
    """log10(raw+1) / scale, clamped to [0,1]."""
    return _sc(math.log10(max(raw, 0) + 1) / scale)


def build_state_vector(result: dict) -> "_np.ndarray":
    """
    Convert an automl_service result dict into a STATE_DIM-dimensional
    float32 numpy array.  All values are in [0, 1].

    This is the single source of truth for the RL state representation —
    feature_extractor.py delegates here so both files stay in sync.

    Parameters
    ----------
    result : dict
        Raw dict from run_automl() or run_automl_extended().

    Returns
    -------
    np.ndarray of shape (STATE_DIM,) = (20,), dtype float32.
    """
    perf = result.get("performance", {}) or {}
    fs   = (result.get("feature_selection") or perf.get("feature_selection") or {})
    diag = result.get("dataset_diagnostics", {}) or {}
    ah   = result.get("agent_history", []) or []
    ba   = result.get("baseline_alert", {}) or {}

    # ── [ 0] n_rows_norm ──────────────────────────────────────────────────────
    n_rows = _sv(diag.get("n_rows") or perf.get("n_train") or perf.get("n_test") or 1000)
    d0 = _ln(n_rows, 7.0)

    # ── [ 1] n_feats_norm ─────────────────────────────────────────────────────
    n_feats = _sv(fs.get("final_features") or diag.get("n_cols") or 10)
    d1 = _ln(n_feats, 4.0)

    # ── [ 2] missing_ratio ────────────────────────────────────────────────────
    d2 = _sc(_sv(diag.get("overall_missing_pct", 0.0)) / 100.0)

    # ── [ 3] imbalance ────────────────────────────────────────────────────────
    # Use majority-class fraction if available; 0.5 = perfectly balanced / regression
    imbalance_raw = diag.get("class_imbalance")       # may be None (regression)
    if imbalance_raw is None:
        # Fallback: derive from n_classes if present
        n_cls = _sv(diag.get("n_classes", 0))
        imbalance_raw = (1.0 / max(n_cls, 1)) if n_cls > 0 else 0.5
    d3 = _sc(_sv(imbalance_raw))

    # ── [ 4] prob_type ────────────────────────────────────────────────────────
    d4 = 1.0 if result.get("problem_type", "classification") == "regression" else 0.0

    # ── [ 5] tier_norm ────────────────────────────────────────────────────────
    tier = _sv(perf.get("scale_tier") or result.get("scale_tier") or 1)
    d5 = _sc(tier / 4.0)

    # ── [ 6] small_dataset ───────────────────────────────────────────────────
    d6 = 1.0 if n_rows < 200 else 0.0

    # ── [ 7] cv_score ─────────────────────────────────────────────────────────
    cv_score = _sv(result.get("cv_score_mean") or perf.get("cv_score_mean") or 0.0)
    d7 = _sc(cv_score)

    # ── [ 8] cv_std_n ─────────────────────────────────────────────────────────
    cv_std = _sv(result.get("cv_score_std") or perf.get("cv_score_std") or 0.1)
    d8 = _sc(cv_std / 0.50)

    # ── [ 9] test_score ───────────────────────────────────────────────────────
    test_score = _sv(
        result.get("test_score")
        or perf.get("accuracy")
        or perf.get("R2")
        or cv_score
    )
    d9 = _sc(max(test_score, 0.0))   # R2 can be negative; clamp at 0

    # ── [10] overfit_gap_n ────────────────────────────────────────────────────
    train_score = _sv(perf.get("train_accuracy") or perf.get("train_R2") or test_score)
    overfit_gap = max(0.0, train_score - cv_score)
    d10 = _sc(overfit_gap / 0.40)

    # ── [11] confidence ───────────────────────────────────────────────────────
    conf = _sv(result.get("confidence_score") or perf.get("confidence_score") or 0.5)
    d11 = _sc(conf)

    # ── [12] leakage ──────────────────────────────────────────────────────────
    d12 = 1.0 if bool(result.get("leakage_detected", False)) else 0.0

    # ── [13] baseline_gap_n ───────────────────────────────────────────────────
    # signed gap: positive = model beats baseline, negative = below baseline
    # shifted to [0,1]: 0.5 means exactly at baseline
    gap_raw = _sv(ba.get("gap", 0.0))
    d13 = _sc(0.5 + gap_raw)

    # ── [14] feat_reduction ───────────────────────────────────────────────────
    orig_f  = _sv(fs.get("original_features") or n_feats)
    final_f = _sv(fs.get("final_features") or n_feats)
    d14 = _sc(1.0 - (final_f / max(orig_f, 1)))

    # ── [15] model_family_n ───────────────────────────────────────────────────
    model_name = str(result.get("best_model") or result.get("best_model_name") or "")
    d15 = _sc(_family_index(model_name) / 4.0)

    # ── [16] stacking ────────────────────────────────────────────────────────
    d16 = 1.0 if "Stacking" in model_name else 0.0

    # ── [17] fs_steps_n ──────────────────────────────────────────────────────
    steps = fs.get("steps_applied") or []
    d17 = _sc(len(steps) / 5.0)

    # ── [18] n_iters_n ────────────────────────────────────────────────────────
    real_iters = [e for e in ah if isinstance(e, dict) and "score" in e]
    d18 = _sc(len(real_iters) / 30.0)

    # ── [19] avg_eff_score ───────────────────────────────────────────────────
    eff_scores = [e["effective_score"] for e in real_iters
                  if e.get("effective_score") is not None]
    if eff_scores:
        d19 = _sc(float(_np.mean(eff_scores)))
    else:
        d19 = d7   # fallback to cv_score when no RL history

    vec = _np.array([
        d0, d1, d2,  d3,  d4,  d5,  d6,
        d7, d8, d9,  d10, d11,
        d12, d13, d14, d15, d16, d17, d18, d19,
    ], dtype=_np.float32)

    assert vec.shape == (STATE_DIM,), f"build_state_vector: shape {vec.shape} ≠ ({STATE_DIM},)"
    return vec