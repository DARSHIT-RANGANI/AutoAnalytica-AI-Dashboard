/**
 * FILE:   src/services/api.js
 * STATUS: NEW
 *
 * Clean API service layer for AutoAnalytica AI backend (v5.5).
 * All fetch calls go through here so Upload.js never contains raw URLs.
 *
 * Usage
 * ─────
 *   import { uploadDataset, trainModel, getDashboard,
 *            predictModel, sendFeedback, getAgentStatus } from "../services/api";
 */

const BASE = "http://127.0.0.1:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────

async function post(path, body, isForm = false) {
  const opts = {
    method: "POST",
    body: isForm ? body : JSON.stringify(body),
  };
  if (!isForm) opts.headers = { "Content-Type": "application/json" };
  const res  = await fetch(`${BASE}${path}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);
  return data;
}

async function get(path) {
  const res  = await fetch(`${BASE}${path}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);
  return data;
}

// ── 1. Upload dataset ─────────────────────────────────────────────────────────

/**
 * Upload a raw CSV / Excel file.
 * Returns { cleaned_filename, analysis: { column_types } }
 */
export async function uploadDataset(file) {
  const fd = new FormData();
  fd.append("file", file);
  return post("/upload/api/upload", fd, true);
}

// ── 2. Train model (v5.5 — returns full agent fields) ────────────────────────

/**
 * Trigger AutoML + RL agent training on an uploaded dataset.
 *
 * Backend returns the full v5.5 payload:
 *   Standard AutoML fields  (performance, best_model, model_quality, …)
 *   + Agent fields          (decision, model_quality_score, meta_insight,
 *                            active_risks, improvements, retrain_decision,
 *                            retrain_probability, retrain_urgency,
 *                            learning_state, workflow_selected,
 *                            drift_analysis, rl_reward, …)
 */
export async function trainModel(filename, targetColumn) {
  return post("/ai/train", {
    filename:      filename,
    target_column: targetColumn,
  });
}

// ── 3. Generate analytics dashboard ─────────────────────────────────────────

/**
 * Generate the visual analytics dashboard for a cleaned file.
 * Returns { charts: [filename, …] }
 */
export async function getDashboard(cleanedFilename) {
  return get(`/dashboard/${cleanedFilename}`);
}

// ── 4. Run prediction ─────────────────────────────────────────────────────────

/**
 * Run a single prediction using a trained model.
 * Returns { prediction, confidence, problem_type, raw_value }
 */
export async function predictModel(modelName, inputData) {
  return post("/ai/predict", {
    model_name: modelName,
    input_data: inputData,
  });
}

// ── 5. Send feedback (close RL reward loop) ───────────────────────────────────

/**
 * Record a ground-truth outcome for a previous training run.
 * This feeds the RL agent reward signal.
 *
 * @param {string}  runId           — agent_run_id from trainModel() response
 * @param {boolean} correct         — was the prediction correct?
 * @param {number}  [newCvScore]    — updated CV score after any retraining
 * @param {boolean} [retrainHelped] — did retraining improve performance?
 */
export async function sendFeedback(runId, correct, newCvScore = null, retrainHelped = null) {
  return post("/ai/feedback", {
    run_id:          runId,
    correct,
    new_cv_score:    newCvScore,
    retrain_helped:  retrainHelped,
  });
}

// ── 6. Agent status health check ─────────────────────────────────────────────

/**
 * Fetch the current learning state of all AI agents.
 * Returns { agents_available, rl_agent, meta_model, retrain_model,
 *           planner, executor }
 */
export async function getAgentStatus() {
  return get("/ai/agents/status");
}

// ── 7. Save agent states ──────────────────────────────────────────────────────

/**
 * Manually persist all agent model weights to disk.
 */
export async function saveAgents() {
  return post("/ai/agents/save", {});
}

// ── 8. List saved models ──────────────────────────────────────────────────────

export async function listModels() {
  return get("/ai/models");
}

// ── 9. Delete a saved model ───────────────────────────────────────────────────

export async function deleteModel(modelId) {
  const res  = await fetch(`${BASE}/ai/models/${modelId}`, { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
  return data;
}

// ── Field extractor — normalises the v5.5 response into a flat object ─────────

/**
 * extractAgentFields(trainResponse)
 *
 * Safely extracts every v5.5 agent field from the raw backend response.
 * Returns defaults for all fields so the UI never crashes on undefined.
 *
 * Call this once after trainModel() and spread the result into state:
 *   const agent = extractAgentFields(res);
 *   setDecision(agent.decision);
 *   …
 */
export function extractAgentFields(r = {}) {
  const ls = r.learning_state || {};
  const da = r.drift_analysis  || {};
  const ba = r.baseline_alert  || {};
  const dd = r.dataset_diagnostics || {};

  return {
    // ── RL agent ──────────────────────────────────────────────────────────────
    decision:            r.decision            || "accept_prediction",
    rl_reward:           r.rl_reward           ?? null,

    // ── Meta-model ────────────────────────────────────────────────────────────
    model_quality_score: r.model_quality_score ?? null,
    model_quality_label: r.model_quality_label || "",
    meta_insight:        r.meta_insight        || "",
    active_risks:        Array.isArray(r.active_risks) ? r.active_risks : [],
    improvements:        Array.isArray(r.improvements) ? r.improvements : [],

    // ── Retrain model ─────────────────────────────────────────────────────────
    retrain_decision:    r.retrain_decision    ?? false,
    retrain_probability: r.retrain_probability ?? null,
    retrain_urgency:     r.retrain_urgency     || "None",

    // ── Agent system ──────────────────────────────────────────────────────────
    workflow_selected:   r.workflow_selected   || "",
    pipeline_version:    r.pipeline_version    || "",
    agent_run_id:        r.agent_run_id        || "",
    agent_total_ms:      r.agent_total_ms      ?? null,

    // ── Drift analysis ────────────────────────────────────────────────────────
    drift_severity:      da.drift_severity     ?? 0,
    rolling_drift:       da.rolling_drift      ?? 0,
    decay_velocity:      da.decay_velocity     ?? 0,
    runs_since_retrain:  da.runs_since_retrain ?? null,

    // ── Baseline alert ────────────────────────────────────────────────────────
    baseline_gap:        ba.gap                ?? null,
    baseline_triggered:  ba.triggered          ?? false,

    // ── Dataset diagnostics ───────────────────────────────────────────────────
    class_imbalance:     dd.class_imbalance    ?? null,
    overall_missing_pct: dd.overall_missing_pct ?? null,
    most_skewed:         dd.most_skewed_features || {},

    // ── Learning state ────────────────────────────────────────────────────────
    ls_epsilon:          ls.epsilon            ?? null,
    ls_step_count:       ls.step_count         ?? null,
    ls_buffer_size:      ls.buffer_size        ?? null,
    ls_avg_reward:       ls.avg_reward_last100 ?? null,
    ls_network_fitted:   ls.network_fitted     ?? false,
    ls_agents_available: r.learning_state
      ? (r.pipeline_version !== "v5.4-no-agents")
      : false,
  };
}

// ── Decision formatting helpers ───────────────────────────────────────────────

export const DECISION_META = {
  accept_prediction: { label: "Accept",        emoji: "✓",  color: "#10b981", bg: "#d1fae5", border: "#6ee7b7" },
  reject_prediction: { label: "Reject",        emoji: "✕",  color: "#ef4444", bg: "#fee2e2", border: "#fca5a5" },
  retrain_model:     { label: "Retrain",       emoji: "🔄", color: "#f59e0b", bg: "#fef3c7", border: "#fcd34d" },
  request_more_data: { label: "Need More Data",emoji: "📥", color: "#6366f1", bg: "#e0e7ff", border: "#a5b4fc" },
};

export function getDecisionMeta(decision = "") {
  const key = decision.toLowerCase().replace(/ /g, "_");
  return DECISION_META[key] || DECISION_META.accept_prediction;
}

export const URGENCY_META = {
  Immediate: { color: "#ef4444", bg: "#fee2e2", label: "Immediate" },
  Soon:      { color: "#f59e0b", bg: "#fef3c7", label: "Soon"      },
  Monitor:   { color: "#6366f1", bg: "#e0e7ff", label: "Monitor"   },
  None:      { color: "#10b981", bg: "#d1fae5", label: "None"      },
};

export function getUrgencyMeta(urgency = "None") {
  return URGENCY_META[urgency] || URGENCY_META.None;
}

export const QUALITY_LABEL_META = {
  Excellent: { color: "#065f46", bg: "#d1fae5", border: "#6ee7b7" },
  Good:      { color: "#1e40af", bg: "#dbeafe", border: "#93c5fd" },
  Fair:      { color: "#92400e", bg: "#fef3c7", border: "#fcd34d" },
  Poor:      { color: "#991b1b", bg: "#fee2e2", border: "#fca5a5" },
};

export function getQualityLabelMeta(label = "Good") {
  return QUALITY_LABEL_META[label] || QUALITY_LABEL_META.Good;
}

export const RISK_META = {
  unstable_cv:               { label: "Unstable CV",       emoji: "📉", color: "#f59e0b" },
  overfitting:               { label: "Overfitting",        emoji: "⚠️", color: "#ef4444" },
  insufficient_data:         { label: "Insufficient Data",  emoji: "📂", color: "#6366f1" },
  near_baseline_performance: { label: "Near Baseline",      emoji: "📊", color: "#8b5cf6" },
};

export function getRiskMeta(riskKey = "") {
  return RISK_META[riskKey] || { label: riskKey.replace(/_/g, " "), emoji: "⚡", color: "#64748b" };
}

export const IMPROVE_META = {
  collect_more_data:           { label: "Collect More Data",       emoji: "📥" },
  engineer_more_features:      { label: "Engineer Features",       emoji: "🔧" },
  increase_regularisation:     { label: "Increase Regularisation", emoji: "🎛️" },
  balance_class_distribution:  { label: "Balance Classes",         emoji: "⚖️" },
  reduce_feature_count:        { label: "Reduce Features",         emoji: "✂️" },
  use_stacking_ensemble:       { label: "Use Ensemble",            emoji: "🏗️" },
  expand_hyperparameter_search:{ label: "Expand HP Search",        emoji: "🔍" },
  improve_data_quality:        { label: "Improve Data Quality",    emoji: "🧹" },
};

export function getImproveMeta(key = "") {
  return IMPROVE_META[key] || { label: key.replace(/_/g, " "), emoji: "💡" };
}