"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  experience_store.py  — v1.0  (AutoAnalytica v5.5)                         ║
║                                                                              ║
║  Shared Experience Store — Global Learning Loop                              ║
║                                                                              ║
║  Problem solved                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Each agent (RL, Meta, Retrain, Distillation) previously stored its own     ║
║  private history. Decisions and rewards were siloed — the retrain model     ║
║  could not benefit from what the distillation policy learned, and the RL    ║
║  agent could not see meta-model confidence signals.                          ║
║                                                                              ║
║  Solution                                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  A single append-only Experience Store that every agent reads and writes.   ║
║  Each entry is a complete Experience tuple:                                  ║
║                                                                              ║
║    {                                                                         ║
║      run_id      : str       — pipeline run identifier                      ║
║      ts          : str       — UTC ISO timestamp                            ║
║      agent       : str       — which agent generated this experience        ║
║      state       : dict      — raw state features at decision time          ║
║      action      : str       — decision taken                               ║
║      outcome     : dict      — observed result (cv_score, test_score, etc.) ║
║      reward      : float     — scalar signal ∈ [0, 1]                      ║
║      uncertainty : dict      — epistemic + aleatoric from feature extractor ║
║      meta        : dict      — arbitrary extra context                      ║
║    }                                                                         ║
║                                                                              ║
║  Consumer API                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  store = SharedExperienceStore.get()          # process singleton           ║
║  store.record(experience)                     # append one entry            ║
║  store.query(agent=…, min_reward=…, n=…)      # filtered retrieval         ║
║  store.build_training_set(agent)              # (X, y) for sklearn          ║
║  store.reward_stats(agent)                    # mean/std/count rewards      ║
║  store.best_actions(state_bucket)             # top actions by avg reward   ║
║                                                                              ║
║  All consumers receive the SAME experiences — cross-agent learning is       ║
║  therefore automatic: distillation rewards influence the RL agent's state   ║
║  representation, meta insights influence retrain decisions, etc.            ║
║                                                                              ║
║  Persistence                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  experience_store/experiences.jsonl  — append-only JSONL log                ║
║  experience_store/store_meta.json    — stats, last compaction ts            ║
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
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_STORE_DIR      = Path("experience_store")
_EXPERIENCES_PATH = _STORE_DIR / "experiences.jsonl"
_STORE_META_PATH  = _STORE_DIR / "store_meta.json"

_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT NAMES  (constants used as the `agent` field)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_RL          = "rl_agent"
AGENT_META        = "meta_model"
AGENT_RETRAIN     = "retrain_model"
AGENT_DISTILL     = "distillation"
AGENT_PIPELINE    = "pipeline"
AGENT_SYSTEM      = "agent_system"

ALL_AGENTS = {AGENT_RL, AGENT_META, AGENT_RETRAIN, AGENT_DISTILL,
              AGENT_PIPELINE, AGENT_SYSTEM}


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def make_experience(
    run_id:      str,
    agent:       str,
    state:       Dict[str, Any],
    action:      str,
    outcome:     Dict[str, Any],
    reward:      float,
    uncertainty: Optional[Dict[str, float]] = None,
    meta:        Optional[Dict[str, Any]]   = None,
) -> Dict:
    """
    Construct a validated experience dict.

    Parameters
    ──────────
    run_id      pipeline run ID (from agent_system or uuid)
    agent       one of the AGENT_* constants
    state       raw numeric state dict (from feature_extractor or rl_agent)
    action      string decision taken
    outcome     result dict (cv_score, test_score, confidence, etc.)
    reward      scalar ∈ [0, 1]
    uncertainty epistemic/aleatoric/confidence_gap from ConfidenceAnalyser
    meta        free-form extra context (e.g. distillation student name)
    """
    reward = float(max(0.0, min(1.0, reward)))
    return {
        "run_id":      run_id,
        "ts":          _ts(),
        "agent":       agent,
        "state":       _safe_dict(state),
        "action":      str(action),
        "outcome":     _safe_dict(outcome),
        "reward":      round(reward, 6),
        "uncertainty": _safe_dict(uncertainty or {}),
        "meta":        _safe_dict(meta or {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED EXPERIENCE STORE
# ─────────────────────────────────────────────────────────────────────────────

class SharedExperienceStore:
    """
    Append-only shared experience store used by every agent in the stack.

    Architecture
    ────────────
    • In-memory ring buffer capped at MAX_MEMORY entries for fast queries
    • Disk persistence via append-only JSONL file (survives restarts)
    • Optional compaction: rewrites JSONL keeping only last MAX_DISK entries

    Cross-agent learning
    ────────────────────
    All agents append to and read from the same store, enabling:

    RL agent       → reads meta/retrain agent outcomes to enrich state
    Meta model     → reads RL agent decisions + rewards for pattern learning
    Retrain model  → reads distillation + pipeline outcomes for drift signals
    Distillation   → reads pipeline outcomes to calibrate policy
    """

    MAX_MEMORY = 2_000    # in-memory ring buffer size
    MAX_DISK   = 10_000   # disk entries before compaction
    COMPACT_EVERY = 1_000 # compact after this many new appends since last compact

    # ── Singleton ─────────────────────────────────────────────────────────────

    _instance: Optional["SharedExperienceStore"] = None

    @classmethod
    def get(cls) -> "SharedExperienceStore":
        """Return the process-level singleton, loading from disk on first call."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard singleton (use after retrains / in tests)."""
        cls._instance = None
        log.info("[ExperienceStore] Singleton reset.")

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._buffer:       List[Dict] = []   # in-memory ring buffer
        self._total_written: int       = 0    # total entries ever written
        self._since_compact: int       = 0    # appends since last compaction
        self._fh           = None             # open JSONL file handle

    # ── Public write API ──────────────────────────────────────────────────────

    def record(self, experience: Dict) -> None:
        """
        Append one experience to the store.

        Thread-safety note: single-process safe. For multi-worker deployments
        use a lock or switch to a database backend.
        """
        # Validate minimal schema
        for required in ("run_id", "agent", "action", "reward"):
            if required not in experience:
                log.warning(f"[ExperienceStore] Missing field '{required}' — recording anyway.")

        entry = dict(experience)
        if "ts" not in entry:
            entry["ts"] = _ts()

        # In-memory ring buffer
        self._buffer.append(entry)
        if len(self._buffer) > self.MAX_MEMORY:
            self._buffer = self._buffer[-self.MAX_MEMORY:]

        # Disk persistence
        try:
            self._append_to_disk(entry)
        except Exception as exc:
            log.warning(f"[ExperienceStore] Disk write failed: {exc}")

        self._total_written  += 1
        self._since_compact  += 1

        if self._since_compact >= self.COMPACT_EVERY:
            self._compact_if_needed()

        log.debug(f"[ExperienceStore] +1 entry "
                  f"agent={entry.get('agent')} action={entry.get('action')} "
                  f"reward={entry.get('reward'):.3f}")

    def record_many(self, experiences: List[Dict]) -> None:
        for exp in experiences:
            self.record(exp)

    # ── Public read API ───────────────────────────────────────────────────────

    def query(
        self,
        agent:      Optional[str]   = None,
        action:     Optional[str]   = None,
        min_reward: float           = -1.0,
        max_reward: float           = 2.0,
        n:          Optional[int]   = None,
        run_id:     Optional[str]   = None,
    ) -> List[Dict]:
        """
        Filter experiences from the in-memory buffer.

        Returns the most recent matching entries (reversed chronologically
        if n is specified).
        """
        results = []
        for entry in reversed(self._buffer):
            if agent      is not None and entry.get("agent")  != agent:          continue
            if action     is not None and entry.get("action") != action:         continue
            if run_id     is not None and entry.get("run_id") != run_id:         continue
            r = entry.get("reward", 0.0)
            if r < min_reward or r > max_reward:                                 continue
            results.append(entry)
            if n is not None and len(results) >= n:
                break
        return list(reversed(results))   # chronological order

    def all(self, n: Optional[int] = None) -> List[Dict]:
        """Return all (or last n) buffered experiences in chronological order."""
        if n is None:
            return list(self._buffer)
        return self._buffer[-n:]

    def reward_stats(self, agent: Optional[str] = None) -> Dict[str, float]:
        """Mean, std, min, max, count of rewards — optionally filtered by agent."""
        entries = self.query(agent=agent) if agent else list(self._buffer)
        if not entries:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        rewards = [e.get("reward", 0.0) for e in entries]
        return {
            "mean":  round(float(np.mean(rewards)),  4),
            "std":   round(float(np.std(rewards)),   4),
            "min":   round(float(np.min(rewards)),   4),
            "max":   round(float(np.max(rewards)),   4),
            "count": len(rewards),
        }

    def best_actions(
        self,
        state_bucket:   Optional[str] = None,
        agent:          Optional[str] = None,
        top_n:          int           = 3,
    ) -> List[Dict]:
        """
        Return the top_n actions ranked by mean reward.

        If state_bucket is given, only entries whose state contains
        the bucket string are considered (coarse matching).
        """
        action_rewards: Dict[str, List[float]] = defaultdict(list)
        for entry in self._buffer:
            if agent is not None and entry.get("agent") != agent:
                continue
            if state_bucket is not None:
                state_str = str(entry.get("state", ""))
                if state_bucket not in state_str:
                    continue
            action_rewards[entry.get("action", "unknown")].append(
                entry.get("reward", 0.0))

        ranked = sorted(
            [{"action": a, "mean_reward": round(float(np.mean(rs)), 4),
              "count": len(rs), "std": round(float(np.std(rs)), 4)}
             for a, rs in action_rewards.items()],
            key=lambda x: x["mean_reward"],
            reverse=True,
        )
        return ranked[:top_n]

    def build_training_set(
        self,
        agent:       Optional[str] = None,
        feature_keys: Optional[List[str]] = None,
        label_key:   str = "reward",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build a (X, y) sklearn-ready training set from stored experiences.

        Parameters
        ──────────
        agent        filter by agent name
        feature_keys list of state/outcome keys to use as features.
                     If None, uses all numeric keys in state + uncertainty.
        label_key    "reward" (default) or any outcome key

        Returns
        ───────
        (X, y) numpy arrays, or (None, None) if insufficient data.
        """
        entries = self.query(agent=agent) if agent else list(self._buffer)
        if len(entries) < 5:
            return None, None

        def _extract_features(entry: Dict) -> Optional[List[float]]:
            state   = entry.get("state", {}) or {}
            outcome = entry.get("outcome", {}) or {}
            uncert  = entry.get("uncertainty", {}) or {}
            combined = {**state, **outcome, **uncert}

            if feature_keys is not None:
                row = []
                for k in feature_keys:
                    v = combined.get(k, 0.0)
                    try:
                        fv = float(v)
                        row.append(0.0 if math.isnan(fv) or math.isinf(fv) else fv)
                    except (TypeError, ValueError):
                        row.append(0.0)
                return row

            # Auto-extract all numeric values
            row = []
            for v in combined.values():
                try:
                    fv = float(v)
                    row.append(0.0 if math.isnan(fv) or math.isinf(fv) else fv)
                except (TypeError, ValueError):
                    pass
            return row if row else None

        def _extract_label(entry: Dict) -> Optional[float]:
            if label_key == "reward":
                return entry.get("reward")
            return entry.get("outcome", {}).get(label_key)

        rows, labels = [], []
        for entry in entries:
            features = _extract_features(entry)
            label    = _extract_label(entry)
            if features is not None and label is not None:
                try:
                    lf = float(label)
                    if not (math.isnan(lf) or math.isinf(lf)):
                        rows.append(features)
                        labels.append(lf)
                except (TypeError, ValueError):
                    pass

        if len(rows) < 5:
            return None, None

        # Normalise row lengths (pad shorter rows with 0)
        max_len = max(len(r) for r in rows)
        rows    = [r + [0.0] * (max_len - len(r)) for r in rows]

        return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)

    def cross_agent_features(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve all experiences for a given run_id across ALL agents,
        and flatten them into a single feature dict.

        This is the key cross-agent learning primitive: allows any model to
        see what every other agent decided and how it turned out.
        """
        run_entries = self.query(run_id=run_id)
        features: Dict[str, Any] = {}
        for entry in run_entries:
            agent  = entry.get("agent", "unknown")
            reward = entry.get("reward", 0.0)
            action = entry.get("action", "")
            uncert = entry.get("uncertainty", {}) or {}

            features[f"{agent}_action"]      = action
            features[f"{agent}_reward"]      = reward
            features[f"{agent}_epistemic"]   = uncert.get("epistemic_uncertainty", 0.0)
            features[f"{agent}_aleatoric"]   = uncert.get("aleatoric_uncertainty", 0.0)
            features[f"{agent}_confidence"]  = uncert.get("calibrated_confidence", 0.5)
        return features

    # ── Stats & reporting ─────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Overview stats for /agent_status endpoint."""
        agent_counts: Dict[str, int] = defaultdict(int)
        for entry in self._buffer:
            agent_counts[entry.get("agent", "unknown")] += 1

        return {
            "total_in_memory":  len(self._buffer),
            "total_written":    self._total_written,
            "agent_breakdown":  dict(agent_counts),
            "reward_stats":     self.reward_stats(),
            "best_actions_all": self.best_actions(top_n=5),
            "store_path":       str(_EXPERIENCES_PATH),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load last MAX_MEMORY entries from disk into the in-memory buffer."""
        if not _EXPERIENCES_PATH.exists():
            log.info("[ExperienceStore] No existing store found — starting fresh.")
            return
        try:
            entries = []
            with open(_EXPERIENCES_PATH) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            self._buffer = entries[-self.MAX_MEMORY:]
            self._total_written = len(entries)
            log.info(f"[ExperienceStore] Loaded {len(entries)} entries from disk "
                     f"({len(self._buffer)} in memory).")
        except Exception as exc:
            log.warning(f"[ExperienceStore] Load failed: {exc}")

    def _append_to_disk(self, entry: Dict) -> None:
        """Append one entry to the JSONL file."""
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_EXPERIENCES_PATH, "a") as fh:
            fh.write(json.dumps(_safe_dict(entry), default=_json_default) + "\n")

    def _compact_if_needed(self) -> None:
        """
        Rewrite the JSONL file keeping only the last MAX_DISK entries.
        Called automatically after COMPACT_EVERY new writes.
        """
        self._since_compact = 0
        try:
            if not _EXPERIENCES_PATH.exists():
                return
            entries = []
            with open(_EXPERIENCES_PATH) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if len(entries) <= self.MAX_DISK:
                return
            entries = entries[-self.MAX_DISK:]
            with open(_EXPERIENCES_PATH, "w") as fh:
                for e in entries:
                    fh.write(json.dumps(_safe_dict(e), default=_json_default) + "\n")
            log.info(f"[ExperienceStore] Compacted to {len(entries)} entries.")
        except Exception as exc:
            log.warning(f"[ExperienceStore] Compaction failed: {exc}")

    def save_meta(self) -> None:
        """Persist store stats to store_meta.json."""
        try:
            _STORE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_STORE_META_PATH, "w") as fh:
                json.dump({
                    "total_written": self._total_written,
                    "buffer_size":   len(self._buffer),
                    "last_saved":    _ts(),
                    "summary":       self.summary(),
                }, fh, indent=2, default=_json_default)
        except Exception as exc:
            log.warning(f"[ExperienceStore] Meta save failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS  (module-level wrappers for the singleton)
# ─────────────────────────────────────────────────────────────────────────────

def record_experience(
    run_id:      str,
    agent:       str,
    state:       Dict,
    action:      str,
    outcome:     Dict,
    reward:      float,
    uncertainty: Optional[Dict] = None,
    meta:        Optional[Dict] = None,
) -> None:
    """Record one experience to the global store. Safe to call from anywhere."""
    exp = make_experience(run_id, agent, state, action, outcome, reward,
                          uncertainty, meta)
    SharedExperienceStore.get().record(exp)


def query_experiences(
    agent:      Optional[str] = None,
    min_reward: float         = -1.0,
    n:          Optional[int] = None,
) -> List[Dict]:
    """Query the global store."""
    return SharedExperienceStore.get().query(agent=agent, min_reward=min_reward, n=n)


def get_best_action(
    state_bucket: str,
    agent:        Optional[str] = None,
    fallback:     str           = "NO_OVERRIDE",
) -> str:
    """
    Return the historically best action for a given state bucket.
    Returns `fallback` (default: "NO_OVERRIDE") when no history exists.
    This is the safe fallback — NOT random.
    """
    best = SharedExperienceStore.get().best_actions(
        state_bucket=state_bucket, agent=agent, top_n=1)
    if best and best[0]["count"] >= 3:   # need at least 3 observations
        return best[0]["action"]
    return fallback


def get_cross_agent_features(run_id: str) -> Dict:
    """Get cross-agent feature dict for a completed run."""
    return SharedExperienceStore.get().cross_agent_features(run_id)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_safe_dict(v) for v in d]
    if isinstance(d, float):
        return None if (math.isnan(d) or math.isinf(d)) else d
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        v = float(d); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(d, np.ndarray):
        return _safe_dict(d.tolist())
    return d


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    import tempfile
    print("\n── experience_store.py self-tests ──")

    # ── 1. make_experience schema ─────────────────────────────────────────────
    exp = make_experience(
        run_id="test_001", agent=AGENT_RL,
        state={"rows": 1000, "features": 20},
        action="BOOST_ENSEMBLE",
        outcome={"cv_score": 0.88, "test_score": 0.86},
        reward=0.84,
        uncertainty={"epistemic_uncertainty": 0.12, "aleatoric_uncertainty": 0.08},
    )
    assert exp["agent"]  == AGENT_RL
    assert exp["reward"] == 0.84
    assert "ts"          in exp
    print(f"✓ make_experience schema OK  keys={list(exp.keys())}")

    # ── 2. record + query ─────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        global _EXPERIENCES_PATH, _STORE_META_PATH, _STORE_DIR
        _orig = (_EXPERIENCES_PATH, _STORE_META_PATH, _STORE_DIR)
        _STORE_DIR        = Path(tmp)
        _EXPERIENCES_PATH = _STORE_DIR / "experiences.jsonl"
        _STORE_META_PATH  = _STORE_DIR / "store_meta.json"
        SharedExperienceStore.reset()

        store = SharedExperienceStore.get()

        for i in range(10):
            store.record(make_experience(
                run_id  = f"run_{i}",
                agent   = AGENT_RL if i % 2 == 0 else AGENT_META,
                state   = {"rows": 1000 + i * 100},
                action  = "BOOST_ENSEMBLE" if i < 5 else "NO_OVERRIDE",
                outcome = {"cv_score": 0.80 + i * 0.01},
                reward  = 0.80 + i * 0.01,
            ))

        assert len(store._buffer) == 10
        rl_entries = store.query(agent=AGENT_RL)
        assert len(rl_entries) == 5
        high_reward = store.query(min_reward=0.85)
        assert len(high_reward) == 5
        print(f"✓ record/query OK  total={len(store._buffer)}, rl={len(rl_entries)}")

        # ── 3. reward_stats ───────────────────────────────────────────────────
        stats = store.reward_stats(agent=AGENT_RL)
        assert stats["count"] == 5
        assert stats["mean"] > 0
        print(f"✓ reward_stats OK  {stats}")

        # ── 4. best_actions ───────────────────────────────────────────────────
        best = store.best_actions(top_n=2)
        assert len(best) <= 2
        assert all("action" in b and "mean_reward" in b for b in best)
        print(f"✓ best_actions OK  {best}")

        # ── 5. build_training_set ─────────────────────────────────────────────
        X, y = store.build_training_set(agent=AGENT_RL)
        assert X is not None and y is not None
        assert X.shape[0] == 5
        assert y.shape[0] == 5
        print(f"✓ build_training_set OK  X={X.shape}  y={y.shape}")

        # ── 6. cross_agent_features ───────────────────────────────────────────
        store.record(make_experience(
            run_id="shared_run", agent=AGENT_RL,
            state={}, action="BOOST_ENSEMBLE",
            outcome={"cv_score": 0.88}, reward=0.85,
            uncertainty={"epistemic_uncertainty": 0.10, "calibrated_confidence": 0.82}))
        store.record(make_experience(
            run_id="shared_run", agent=AGENT_META,
            state={}, action="stacking_wins",
            outcome={"cv_score": 0.88}, reward=0.87))

        cross = store.cross_agent_features("shared_run")
        assert f"{AGENT_RL}_action"   in cross
        assert f"{AGENT_META}_reward" in cross
        print(f"✓ cross_agent_features OK  keys={list(cross.keys())}")

        # ── 7. get_best_action fallback ───────────────────────────────────────
        action = get_best_action("unknown_bucket_xyz", fallback="NO_OVERRIDE")
        assert action == "NO_OVERRIDE"
        print(f"✓ get_best_action safe fallback = '{action}'")

        # ── 8. disk persistence round-trip ────────────────────────────────────
        store.save_meta()
        SharedExperienceStore.reset()
        store2 = SharedExperienceStore.get()
        assert len(store2._buffer) > 0
        print(f"✓ Disk persistence round-trip OK  loaded={len(store2._buffer)}")

        # ── 9. summary ────────────────────────────────────────────────────────
        summ = store2.summary()
        assert "agent_breakdown" in summ
        assert "reward_stats"    in summ
        print(f"✓ summary OK  agents={summ['agent_breakdown']}")

        SharedExperienceStore.reset()
        _EXPERIENCES_PATH, _STORE_META_PATH, _STORE_DIR = _orig

    print("\n✓ All experience_store.py self-tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()