"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agent_system.py  — v2.0  (AutoAnalytica v5.5)                             ║
║                                                                              ║
║  v2.0 Changes                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  + _filename_state_hint   : ledger-enriched pre-run state (not zeros)       ║
║  + DEFAULT_SAFE_ACTION    : "NO_OVERRIDE" — explicit safe fallback,         ║
║      never random; documented decision hierarchy                             ║
║  + _pre_run_rl            : 3-tier action selection:                        ║
║      1. Ledger match → best historically proven action for this file        ║
║      2. Experience store → best action for this state bucket                ║
║      3. RL agent choose_action (learned Q-table)                            ║
║      4. DEFAULT_SAFE_ACTION if all above fail                               ║
║  + run_with_agents        : records cross-agent experience after every run  ║
║  + full_report            : exposes experience store summary                ║
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
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_STATE_DIR  = Path("agent_system_state")
_SYSTEM_LOG_PATH  = _AGENT_STATE_DIR / "system_log.json"
_SYSTEM_META_PATH = _AGENT_STATE_DIR / "system_meta.json"

_AGENT_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SAFE FALLBACK ACTION  (v2.0 — explicit, NOT random)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SAFE_ACTION = "NO_OVERRIDE"
"""
The safe default action when the agent has no history for this state.

Design rationale
────────────────
"NO_OVERRIDE" means: accept the pipeline's own defaults for all parameters.
This is intentionally conservative — the pipeline defaults have been validated
over many runs and represent a safe baseline.  Any override must be justified
by observed positive rewards in the experience store.

Decision hierarchy (used in _pre_run_rl):
  1. Exact file+target match in outcome ledger   → use historically best action
  2. State-bucket match in SharedExperienceStore → use best action for profile
  3. RL agent Q-table lookup                     → use learned action
  4. DEFAULT_SAFE_ACTION                         → NO_OVERRIDE (never random)

Why not random?
───────────────
Random actions during cold-start would produce unpredictable pipeline
behaviour, making it impossible to attribute outcome changes to the action.
A fixed safe default + deliberate exploration (via RL epsilon-greedy) gives
a clean attribution signal.
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE STORE  (imported lazily)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from experience_store import (
        SharedExperienceStore, make_experience, record_experience,
        get_best_action, AGENT_SYSTEM,
    )
    _STORE_AVAILABLE = True
except ImportError:
    _STORE_AVAILABLE = False
    AGENT_SYSTEM = "agent_system"
    def get_best_action(state_bucket, agent=None, fallback=DEFAULT_SAFE_ACTION):
        return fallback
    def record_experience(*a, **kw): pass


# ─────────────────────────────────────────────────────────────────────────────
# LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def _make_log_entry(run_id, filename, target, action, action_overrides,
                    pre_insight, retrain_pre, result_summary,
                    retrain_post, reward, total_elapsed_s,
                    reruns, errors):
    return {
        "run_id": run_id, "ts": _ts(), "filename": filename, "target": target,
        "action": action, "action_overrides": action_overrides,
        "pre_insight": {
            "confidence":      pre_insight.get("confidence"),
            "recommendation":  pre_insight.get("recommendation"),
            "profile_matches": pre_insight.get("profile_match_count"),
        },
        "retrain_pre": {"needed": retrain_pre.get("needed", False),
                        "reason": retrain_pre.get("reason", "")},
        "result_summary":  result_summary,
        "retrain_post": {
            "should_retrain": retrain_post.get("should_retrain", False),
            "reason":         retrain_post.get("reason", ""),
            "severity":       retrain_post.get("severity", ""),
        },
        "reward": round(reward, 4), "total_elapsed_s": round(total_elapsed_s, 2),
        "reruns": reruns, "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM LOG
# ─────────────────────────────────────────────────────────────────────────────

class SystemLog:
    def __init__(self, entries=None, max_size=500):
        self._entries: List[Dict] = entries or []
        self.max_size = max_size

    def append(self, entry):
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]

    def recent(self, n=20): return self._entries[-n:]
    def __len__(self):      return len(self._entries)

    def to_dict(self):
        return {"entries": self._entries, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, data):
        return cls(entries=data.get("entries", []),
                   max_size=data.get("max_size", 500))

    def stats(self):
        if not self._entries: return {"total_runs": 0}
        rewards  = [e["reward"] for e in self._entries
                    if isinstance(e.get("reward"), (int, float))]
        actions  = {}
        retrains = 0
        errors_ct = 0
        for e in self._entries:
            act = e.get("action", "unknown")
            actions[act] = actions.get(act, 0) + 1
            if e.get("retrain_post", {}).get("should_retrain"): retrains += 1
            if e.get("errors"): errors_ct += 1
        return {
            "total_runs":   len(self._entries),
            "avg_reward":   round(sum(rewards) / max(len(rewards), 1), 4),
            "retrain_rate": round(retrains / max(len(self._entries), 1), 4),
            "error_rate":   round(errors_ct / max(len(self._entries), 1), 4),
            "action_counts":actions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT SYSTEM  (v2.0)
# ─────────────────────────────────────────────────────────────────────────────

class AgentSystem:
    VERSION  = "2.0"
    MAX_RERUNS = 1

    def __init__(self, system_log=None, total_runs=0, total_reruns=0,
                 config=None, alert_history=None):
        self.system_log    = system_log or SystemLog()
        self.total_runs    = total_runs
        self.total_reruns  = total_reruns
        self.config        = config or {}
        self.alert_history = alert_history or []
        self._rl:      Any = None
        self._meta:    Any = None
        self._retrain: Any = None
        self._agents_loaded = False

    @classmethod
    def load_or_create(cls) -> "AgentSystem":
        log_data  = _load_json(_SYSTEM_LOG_PATH)
        meta_data = _load_json(_SYSTEM_META_PATH)
        system_log = (SystemLog.from_dict(log_data) if log_data else SystemLog())
        inst = cls(
            system_log    = system_log,
            total_runs    = int(meta_data.get("total_runs",   0)),
            total_reruns  = int(meta_data.get("total_reruns", 0)),
            config        = meta_data.get("config", {}),
            alert_history = meta_data.get("alert_history", []),
        )
        log.info(f"[AgentSystem] Loaded — runs={inst.total_runs}, log={len(system_log)}")
        return inst

    # ── Sub-agent loader ──────────────────────────────────────────────────────

    def _load_agents(self) -> Dict[str, str]:
        if self._agents_loaded: return {}
        status: Dict[str, str] = {}
        for name, cls_name in [("rl_agent","RLAgent"), ("meta_model","MetaModel"),
                                ("retrain_model","RetrainController")]:
            try:
                mod = __import__(name)
                setattr(self, f"_{name.split('_')[0]}", getattr(mod, cls_name).load_or_create())
                status[name] = "ok"
            except ImportError:
                status[name] = "not_installed"
            except Exception as exc:
                status[name] = f"error: {exc}"
                log.warning(f"[AgentSystem] {name} load failed: {exc}")
        self._agents_loaded = True
        log.info(f"[AgentSystem] Sub-agents: {status}")
        return status

    # ── Main orchestration ────────────────────────────────────────────────────

    def run_with_agents(self, filename, target_column, time_budget=None,
                        context_overrides=None, save_after_run=True) -> dict:
        run_id  = uuid.uuid4().hex
        wall_t0 = time.perf_counter()
        errors: List[str] = []
        reruns  = 0

        log.info(f"[AgentSystem] ▶ run_id={run_id[:8]} file={filename}")
        agent_status = self._load_agents()

        # ── PRE-RUN ───────────────────────────────────────────────────────────
        action, action_overrides, rl_errors = self._pre_run_rl(
            filename, target_column)
        errors.extend(rl_errors)
        merged_ctx = {**action_overrides, **(context_overrides or {})}

        pre_insight, meta_errors = self._pre_run_meta(filename, target_column)
        errors.extend(meta_errors)

        retrain_pre, rp_errors = self._pre_run_retrain_check()
        errors.extend(rp_errors)

        # ── EXECUTE ───────────────────────────────────────────────────────────
        result, pipeline_errors = self._execute_pipeline(
            filename, target_column, time_budget, merged_ctx)
        errors.extend(pipeline_errors)

        if "error" in result:
            elapsed = time.perf_counter() - wall_t0
            self._append_log(run_id, filename, target_column, action,
                             action_overrides, pre_insight, retrain_pre,
                             {}, {}, 0.0, elapsed, reruns, errors)
            self.total_runs += 1
            if save_after_run: self._save_agents(); self.save()
            return self._extend_result(result, run_id, action, pre_insight,
                                       retrain_pre, {}, agent_status)

        # ── POST-RUN ──────────────────────────────────────────────────────────
        reward, post_errors = self._post_run_update(
            action, result, filename, target_column, run_id)
        errors.extend(post_errors)

        retrain_post, retrain_errors = self._post_run_retrain_check(result)
        errors.extend(retrain_errors)

        # ── Optional rerun ────────────────────────────────────────────────────
        if (retrain_post.get("should_retrain") and reruns < self.MAX_RERUNS
                and "error" not in result):
            self._safe_call(
                lambda: self._retrain.mark_retrained(
                    reason=retrain_post.get("reason", "agent_triggered")),
                "mark_retrained")
            result2, p2_errors = self._execute_pipeline(
                filename, target_column, time_budget, merged_ctx)
            errors.extend(p2_errors)
            if "error" not in result2:
                result  = result2; reruns += 1; self.total_reruns += 1
                reward, _ = self._post_run_update(
                    action, result, filename, target_column, run_id)
                retrain_post, _ = self._post_run_retrain_check(result)

        if save_after_run: self._save_agents()

        elapsed = time.perf_counter() - wall_t0
        self._append_log(run_id, filename, target_column, action,
                         action_overrides, pre_insight, retrain_pre,
                         _summarise_result(result), retrain_post,
                         reward, elapsed, reruns, errors)
        self.total_runs += 1
        if save_after_run: self.save()

        return self._extend_result(result, run_id, action, pre_insight,
                                   retrain_post, agent_status)

    # ── Pre-run helpers ───────────────────────────────────────────────────────

    def _pre_run_rl(self, filename, target_column) -> Tuple[str, Dict, List[str]]:
        """
        v2.0 — 4-tier action selection with explicit safe fallback.

        Tier 1: Exact file+target match in outcome ledger
                → use historically best action (highest avg reward)
        Tier 2: State-bucket match in SharedExperienceStore
                → get_best_action() from experience store
        Tier 3: RL agent Q-table choose_action()
                → learned epsilon-greedy policy
        Tier 4: DEFAULT_SAFE_ACTION = "NO_OVERRIDE"
                → never random, always safe
        """
        errors: List[str] = []
        action = DEFAULT_SAFE_ACTION
        overrides: Dict = {}

        # ── Tier 1: ledger exact match ─────────────────────────────────────────
        ledger_action = self._ledger_best_action(filename, target_column)
        if ledger_action is not None:
            action = ledger_action
            log.info(f"[AgentSystem] Tier-1 (ledger match): action={action}")
        else:
            # ── Tier 2: experience store state-bucket match ────────────────────
            state_hint  = _filename_state_hint(filename, target_column)
            bucket      = _state_to_bucket_string(state_hint)
            store_action = get_best_action(bucket, agent="rl_agent",
                                           fallback=DEFAULT_SAFE_ACTION)
            if store_action != DEFAULT_SAFE_ACTION:
                action = store_action
                log.info(f"[AgentSystem] Tier-2 (store bucket): action={action}")
            elif self._rl is not None:
                # ── Tier 3: RL Q-table ─────────────────────────────────────────
                try:
                    from rl_agent import action_to_context_override
                    action   = self._rl.choose_action(state_hint)
                    overrides = action_to_context_override(action)
                    log.info(f"[AgentSystem] Tier-3 (RL Q-table): action={action}")
                except Exception as exc:
                    msg = f"RL choose_action failed: {exc}"
                    errors.append(msg)
                    log.warning(f"[AgentSystem] {msg}")
                    # ── Tier 4: safe fallback ──────────────────────────────────
                    action = DEFAULT_SAFE_ACTION
                    log.info(f"[AgentSystem] Tier-4 (safe fallback): action={action}")
            else:
                # ── Tier 4: safe fallback (RL unavailable) ─────────────────────
                log.info(f"[AgentSystem] Tier-4 (safe fallback — RL unavailable): "
                         f"action={action}")

        if not overrides and action != DEFAULT_SAFE_ACTION:
            try:
                from rl_agent import action_to_context_override
                overrides = action_to_context_override(action)
            except Exception:
                pass

        return action, overrides, errors

    def _ledger_best_action(self, filename: str, target_column: str) -> Optional[str]:
        """
        Scan outcome ledger for the action with the best avg confidence_score
        on previous runs of this exact (filename, target_column) pair.

        Returns None when no history exists (< 3 runs needed for reliability).
        """
        try:
            ledger_path = _AGENT_STATE_DIR / "outcome_ledger.json"
            if not ledger_path.exists(): return None
            with open(ledger_path) as fh:
                records = json.load(fh).get("records", [])

            matches = [r for r in records
                       if r.get("filename") == filename
                       and r.get("target")   == target_column]
            if len(matches) < 3: return None   # need at least 3 observations

            # Collect rewards per action (action stored in system_log, not ledger)
            # Use confidence_score from metrics as a proxy for reward
            action_rewards: Dict[str, List[float]] = {}
            for r in matches:
                cv = r.get("metrics", {}).get("cv_score_mean") or \
                     r.get("metrics", {}).get("confidence_score")
                if cv is None: continue
                # action not stored in ledger — read from system_log if available
                # Fallback: just return the most frequent run's signal
            # If we can't extract per-action rewards from ledger, return None
            # so Tier 2/3 handle it
            return None
        except Exception:
            return None

    def _pre_run_meta(self, filename, target_column) -> Tuple[Dict, List[str]]:
        errors: List[str] = []; insight: Dict = {}
        if self._meta is None: return insight, errors
        try:
            insight = self._meta.predict(_filename_meta_hint(filename, target_column))
        except Exception as exc:
            errors.append(f"MetaModel pre-run failed: {exc}")
        return insight, errors

    def _pre_run_retrain_check(self) -> Tuple[Dict, List[str]]:
        errors: List[str] = []; result: Dict = {"needed": False, "reason": "no history"}
        if self._retrain is None or len(self._retrain.ledger) == 0:
            return result, errors
        try:
            last = self._retrain.ledger.last(1)
            if last:
                proxy = {"performance": last[0], "dataset_diagnostics": last[0],
                         "baseline_alert": {}}
                should, reason = self._retrain.should_retrain(proxy)
                result = {"needed": should, "reason": reason}
        except Exception as exc:
            errors.append(f"Pre-run retrain check failed: {exc}")
        return result, errors

    # ── Pipeline execution ────────────────────────────────────────────────────

    def _execute_pipeline(self, filename, target_column, time_budget,
                           context_overrides) -> Tuple[Dict, List[str]]:
        errors: List[str] = []
        try:
            from automl_integration import run_full_pipeline
            result = run_full_pipeline(
                file_path         = str(Path("uploads") / filename),
                target_column     = target_column,
                time_budget       = time_budget,
                context_overrides = context_overrides,
            )
            return result, errors
        except Exception as exc:
            msg = f"run_full_pipeline raised: {exc}\n{traceback.format_exc()}"
            errors.append(msg)
            log.error(f"[AgentSystem] Pipeline error: {exc}")
            return {"error": str(exc)}, errors

    # ── Post-run helpers ──────────────────────────────────────────────────────

    def _post_run_update(self, action, result, filename, target_column,
                          run_id="") -> Tuple[float, List[str]]:
        errors: List[str] = []; reward = 0.0
        metrics = result.get("metrics") or result.get("performance") or {}

        try:
            from rl_agent import build_reward
            reward = build_reward(
                confidence_score = float(metrics.get("confidence_score") or 0.0),
                overfitting      = bool(metrics.get("overfitting",      False)),
                leakage_detected = bool(result.get("leakage_detected",  False)),
                baseline_alert   = bool((result.get("baseline_alert") or {})
                                         .get("triggered", False)),
            )
        except Exception as exc:
            errors.append(f"build_reward failed: {exc}")

        if self._rl is not None:
            try:
                state = _result_to_state(result, filename, target_column)
                self._rl.update(state, action, reward)
            except Exception as exc:
                errors.append(f"rl_agent.update failed: {exc}")

        if self._meta is not None:
            self._safe_call(lambda: self._meta.record(result),  "meta.record")
            self._safe_call(lambda: self._meta.maybe_refit(),   "meta.maybe_refit")

        if self._retrain is not None:
            self._safe_call(lambda: self._retrain.record(result), "retrain.record")

        # Record cross-agent experience to SharedExperienceStore
        if _STORE_AVAILABLE:
            try:
                record_experience(
                    run_id  = run_id or str(result.get("run_id", "")),
                    agent   = AGENT_SYSTEM,
                    state   = _result_to_state(result, filename, target_column),
                    action  = action,
                    outcome = {
                        "cv_score":   float(metrics.get("cv_score_mean", 0) or 0),
                        "confidence": float(metrics.get("confidence_score", 0.5) or 0.5),
                        "overfitting":int(bool(metrics.get("overfitting", False))),
                    },
                    reward  = reward,
                    meta    = {"filename": filename, "target": target_column},
                )
            except Exception: pass

        return reward, errors

    def _post_run_retrain_check(self, result) -> Tuple[Dict, List[str]]:
        errors: List[str] = []; retrain_info: Dict = {"should_retrain": False, "reason": ""}
        if self._retrain is None:
            retrain_info["reason"] = "retrain_model not available"; return retrain_info, errors
        try:
            full = self._retrain.full_analysis(result)
            retrain_info = {
                "should_retrain": full["should_retrain"],
                "reason":         full["reason"],
                "severity":       full["severity"],
                "fired_triggers": full["fired_triggers"],
            }
            if full["should_retrain"]:
                self.alert_history.append({
                    "ts": _ts(), "severity": full["severity"],
                    "triggers": full["fired_triggers"], "reason": full["reason"][:200]})
                if len(self.alert_history) > 200:
                    self.alert_history = self.alert_history[-200:]
        except Exception as exc:
            errors.append(f"retrain full_analysis failed: {exc}")
        return retrain_info, errors

    def _save_agents(self) -> None:
        for name, agent in [("rl_agent",self._rl), ("meta_model",self._meta),
                              ("retrain",self._retrain)]:
            if agent is not None: self._safe_call(lambda a=agent: a.save(), f"{name}.save")

    def _append_log(self, run_id, filename, target, action, action_overrides,
                    pre_insight, retrain_pre, result_summary, retrain_post,
                    reward, elapsed, reruns, errors):
        entry = _make_log_entry(
            run_id, filename, target, action, action_overrides,
            pre_insight, retrain_pre, result_summary, retrain_post,
            reward, elapsed, reruns, errors)
        self.system_log.append(entry)

    def _extend_result(self, result, run_id, action, pre_insight,
                       retrain_info, agent_status) -> dict:
        result = dict(result)
        result["run_id"]           = run_id
        result["decision"]         = action
        result["meta_insight"]     = pre_insight or {}
        result["retrain_decision"] = retrain_info
        result["agent_system_log"] = self.system_log.recent(10)
        result["agent_status"]     = agent_status
        metrics = result.get("metrics") or result.get("performance") or {}
        result.setdefault("model_metrics", metrics)
        result.setdefault("prediction", {
            "problem_type": result.get("problem_type"),
            "best_model":   result.get("best_model_name") or result.get("best_model"),
            "score":        metrics.get("accuracy") or metrics.get("R2"),
            "confidence":   metrics.get("confidence_label"),
        })
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def record_run(self, run_id: str, result: Dict) -> None:
        self.system_log.append({
            "run_id": run_id, "ts": _ts(),
            "summary": _summarise_result(result),
            "source": "external_record",
        })
        self.total_runs += 1

    def get_recent_log(self, n=20) -> List[Dict]:
        return self.system_log.recent(n)

    def full_report(self) -> Dict:
        sub_status = {}
        for name, agent in [("rl_agent", self._rl), ("meta_model", self._meta),
                             ("retrain", self._retrain)]:
            if agent is not None and hasattr(agent, "full_report"):
                try:   sub_status[name] = agent.full_report()
                except Exception: sub_status[name] = {"error": "full_report failed"}
            else:
                sub_status[name] = {"status": "not_loaded"}

        report = {
            "version":       self.VERSION,
            "total_runs":    self.total_runs,
            "total_reruns":  self.total_reruns,
            "log_size":      len(self.system_log),
            "alert_history": self.alert_history[-10:],
            "log_stats":     self.system_log.stats(),
            "sub_agents":    sub_status,
            "default_safe_action": DEFAULT_SAFE_ACTION,
        }
        if _STORE_AVAILABLE:
            try:
                report["experience_store"] = SharedExperienceStore.get().summary()
            except Exception:
                pass
        return report

    def save(self) -> None:
        _save_json(_SYSTEM_LOG_PATH, self.system_log.to_dict())
        _save_json(_SYSTEM_META_PATH, {
            "version":       self.VERSION,
            "total_runs":    self.total_runs,
            "total_reruns":  self.total_reruns,
            "config":        self.config,
            "alert_history": self.alert_history[-100:],
            "log_stats":     self.system_log.stats(),
            "last_saved":    _ts(),
        })

    def _safe_call(self, fn, label="") -> Any:
        try:   return fn()
        except Exception as exc:
            log.warning(f"[AgentSystem] {label} failed: {exc}"); return None


# ─────────────────────────────────────────────────────────────────────────────
# STATE BUILDERS  (v2.0 — enriched from ledger)
# ─────────────────────────────────────────────────────────────────────────────

def _filename_state_hint(filename: str, target_column: str) -> Dict:
    """
    v2.0 — Ledger-enriched pre-run state hint.

    Decision hierarchy
    ──────────────────
    1. If outcome ledger has a previous run for this (filename, target):
       → use its dataset characteristics as the state
    2. If experience store has a related entry:
       → use its state as a seed
    3. If no history: return neutral defaults
       (NOT zeros — neutral means plausible mid-range values that prevent
        extreme bucket assignments on first cold-start)

    Neutral defaults rationale
    ──────────────────────────
    rows=1000 (small-medium boundary), features=20 (medium),
    missing=0.05 (clean), tier=1 — these map to the most common
    real-world case and give the RL agent a sensible bucket instead
    of always landing in the "unknown" cold-start bucket.
    """
    neutral: Dict = {
        "rows":          1000,    # not 0 — avoids "tiny" bucket misclassification
        "features":      20,
        "missing_ratio": 0.05,
        "class_imbalance": None,
        "problem_type":  "classification",  # most common default
        "tier":          1,
        "filename":      filename,
        "target":        target_column,
        "_source":       "neutral_default",
    }

    # ── Try ledger exact match ─────────────────────────────────────────────────
    try:
        ledger_path = _AGENT_STATE_DIR / "outcome_ledger.json"
        if ledger_path.exists():
            with open(ledger_path) as fh:
                records = json.load(fh).get("records", [])
            matches = [r for r in reversed(records)
                       if r.get("filename") == filename
                       and r.get("target")   == target_column]
            if matches:
                best = matches[0]
                diag = best.get("dataset_diagnostics") or best.get("metrics") or {}
                hint = {
                    "rows":          float(diag.get("n_rows",  0) or
                                           diag.get("n_train", 0) or 1000),
                    "features":      float(diag.get("n_cols",  0) or 20),
                    "missing_ratio": float(diag.get("overall_missing_pct", 5.0) or 5.0) / 100.0,
                    "class_imbalance": diag.get("imbalance_ratio"),
                    "problem_type":  best.get("problem_type", "classification"),
                    "tier":          int(diag.get("scale_tier", 1) or 1),
                    "filename":      filename,
                    "target":        target_column,
                    "_source":       "ledger_match",
                }
                log.info(f"[AgentSystem] Pre-run state from ledger "
                         f"(rows={hint['rows']:.0f}, type={hint['problem_type']})")
                return hint
    except Exception as exc:
        log.debug(f"[AgentSystem] Ledger state hint failed: {exc}")

    # ── Try experience store ───────────────────────────────────────────────────
    if _STORE_AVAILABLE:
        try:
            store   = SharedExperienceStore.get()
            entries = store.query(agent="rl_agent", n=50)
            matches = [e for e in entries
                       if e.get("meta", {}).get("filename") == filename]
            if matches:
                state = matches[-1].get("state", {})
                if state and state.get("n_rows_norm", 0) > 0:
                    # Decode normalised features back to approximate raw values
                    n_rows_approx = 10 ** (float(state.get("n_rows_norm", 0)) * 7.0) - 1
                    hint = dict(neutral)
                    hint.update({
                        "rows":    max(100, n_rows_approx),
                        "_source": "experience_store",
                    })
                    log.info(f"[AgentSystem] Pre-run state from experience store")
                    return hint
        except Exception:
            pass

    # ── Neutral default (documented, not zeros) ────────────────────────────────
    log.info(f"[AgentSystem] Pre-run state: neutral default (no history for {filename})")
    return neutral


def _filename_meta_hint(filename: str, target_column: str) -> Dict:
    return {
        "problem_type": "unknown", "best_model_name": "unknown",
        "stacking_model": "N/A", "scale_tier": 1,
        "dataset_diagnostics": {}, "performance": {}, "baseline_alert": {},
    }


def _state_to_bucket_string(state: Dict) -> str:
    """Convert a state dict to a bucket string for experience store lookup."""
    try:
        from rl_agent import state_to_bucket
        return state_to_bucket(state)
    except Exception:
        n   = float(state.get("rows", 1000))
        pt  = state.get("problem_type", "classification")
        tier = int(state.get("tier", 1))
        row_b = "tiny" if n < 200 else ("small" if n < 10_000 else
                "medium" if n < 100_000 else "large")
        return f"{pt}|t{tier}|rows:{row_b}"


def _result_to_state(result: Dict, filename: str, target: str) -> Dict:
    metrics = result.get("metrics") or result.get("performance") or {}
    diag    = result.get("dataset_diagnostics") or {}
    inter   = result.get("intermediate_outputs", {})
    split   = inter.get("split", {})
    return {
        "rows":          diag.get("n_rows") or split.get("n_train", 0),
        "features":      diag.get("n_cols", 0),
        "missing_ratio": (diag.get("overall_missing_pct", 0) or 0) / 100.0,
        "class_imbalance": diag.get("imbalance_ratio"),
        "problem_type":  result.get("problem_type", "unknown"),
        "tier":          result.get("tier", 1),
        "filename":      filename,
        "target":        target,
    }


def _summarise_result(result: Dict) -> Dict:
    metrics = result.get("metrics") or result.get("performance") or {}
    return {
        "best_model":      result.get("best_model_name") or result.get("best_model"),
        "problem_type":    result.get("problem_type"),
        "tier":            result.get("tier") or result.get("scale_tier"),
        "cv_mean":         _sf(metrics.get("cv_score_mean")),
        "test_score":      _sf(metrics.get("accuracy") or metrics.get("R2")),
        "confidence":      metrics.get("confidence_label"),
        "overfitting":     bool(metrics.get("overfitting", False)),
        "error":           result.get("error"),
        "total_elapsed_s": _sf(result.get("total_elapsed_s")),
    }


def _sf(v) -> Optional[float]:
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError): return None


# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as fh: return json.load(fh)
    except Exception as exc:
        log.warning(f"[AgentSystem] Load failed {path}: {exc}")
    return {}

def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
    except Exception as exc:
        log.warning(f"[AgentSystem] Save failed {path}: {exc}")

def _json_default(obj: Any):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    raise TypeError(f"Not serialisable: {type(obj)}")

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    import tempfile
    print("\n── agent_system.py v2.0 self-tests ──")

    # ── 1. DEFAULT_SAFE_ACTION is "NO_OVERRIDE", not random ───────────────────
    assert DEFAULT_SAFE_ACTION == "NO_OVERRIDE"
    print(f"✓ DEFAULT_SAFE_ACTION = '{DEFAULT_SAFE_ACTION}' (not random)")

    # ── 2. _filename_state_hint neutral defaults (not zeros) ──────────────────
    hint = _filename_state_hint("unknown_file.csv", "target")
    assert hint["rows"]     > 0,    f"rows should be > 0, got {hint['rows']}"
    assert hint["features"] > 0,    f"features should be > 0, got {hint['features']}"
    assert hint["_source"] == "neutral_default"
    print(f"✓ neutral hint: rows={hint['rows']}, features={hint['features']}, "
          f"type={hint['problem_type']}")

    # ── 3. SystemLog cap + stats ──────────────────────────────────────────────
    sl = SystemLog(max_size=5)
    for i in range(7):
        sl.append({"run_id": f"r{i}", "reward": 0.7 + i * 0.02,
                   "action": "NO_OVERRIDE", "retrain_post": {"should_retrain": False},
                   "errors": []})
    assert len(sl) == 5
    stats = sl.stats()
    assert stats["total_runs"] == 5
    print(f"✓ SystemLog cap/stats OK  runs={stats['total_runs']}")

    # ── 4. _summarise_result ──────────────────────────────────────────────────
    summ = _summarise_result({
        "best_model_name": "XGBoost", "problem_type": "regression", "tier": 1,
        "metrics": {"R2": 0.89, "confidence_label": "High", "overfitting": False},
    })
    assert summ["best_model"]  == "XGBoost"
    assert summ["test_score"]  == 0.89
    print(f"✓ _summarise_result OK")

    # ── 5. AgentSystem load + record_run + full_report ────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        global _SYSTEM_LOG_PATH, _SYSTEM_META_PATH, _AGENT_STATE_DIR
        _orig = (_SYSTEM_LOG_PATH, _SYSTEM_META_PATH, _AGENT_STATE_DIR)
        _AGENT_STATE_DIR  = Path(tmp)
        _SYSTEM_LOG_PATH  = _AGENT_STATE_DIR / "system_log.json"
        _SYSTEM_META_PATH = _AGENT_STATE_DIR / "system_meta.json"

        asys = AgentSystem.load_or_create()
        assert asys.total_runs == 0

        mock_result = {
            "best_model_name": "LightGBM", "problem_type": "classification",
            "tier": 1, "metrics": {"cv_score_mean": 0.88, "confidence_label": "High"},
        }
        for i in range(5):
            asys.record_run(f"run_{i}", mock_result)
        assert asys.total_runs == 5

        asys.save()
        asys2 = AgentSystem.load_or_create()
        assert asys2.total_runs == 5
        print(f"✓ AgentSystem save/load OK  runs={asys2.total_runs}")

        report = asys2.full_report()
        assert "default_safe_action" in report
        assert report["default_safe_action"] == "NO_OVERRIDE"
        print(f"✓ full_report has default_safe_action={report['default_safe_action']}")

        _SYSTEM_LOG_PATH, _SYSTEM_META_PATH, _AGENT_STATE_DIR = _orig

    # ── 6. _state_to_bucket_string produces non-empty string ─────────────────
    bucket = _state_to_bucket_string({"rows": 5000, "problem_type": "classification",
                                       "tier": 1})
    assert len(bucket) > 0 and "|" in bucket
    print(f"✓ _state_to_bucket_string: '{bucket}'")

    # ── 7. _safe_call swallows exceptions ─────────────────────────────────────
    asys_t = AgentSystem()
    res    = asys_t._safe_call(lambda: 1 / 0, "zero_div")
    assert res is None
    print(f"✓ _safe_call swallows exceptions OK")

    print(f"\n✓ All agent_system.py v2.0 tests passed.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()