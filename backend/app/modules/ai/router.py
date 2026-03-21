# =============================================================================
# backend/app/modules/ai/router.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. train_model() is now  async def  so it can await MongoDB insertions.
#    run_automl() (sync / CPU-heavy) is offloaded to a thread-pool executor
#    via asyncio.get_event_loop().run_in_executor() so the event loop is
#    never blocked during long training jobs.
#
# 2. After training, model metadata is inserted into the 'models' collection.
#    The returned payload now includes  "model_id"  (MongoDB _id string).
#
# 3. AutoMLRequest now accepts an optional  dataset_id  field (see
#    app/schemas/automl_schema.py).  If provided, it is stored in the model
#    document for linking.  If not provided, we attempt a filename lookup.
#
# 4. GET  /ai/models          →  list all models from MongoDB
# 5. GET  /ai/models/{id}     →  get a single model by MongoDB _id
# 6. DELETE /ai/models/{id}   →  delete DB record + physical .pkl file
#
# All existing predict / download-report-pdf endpoints are unchanged.
#
# v5.5 AGENT INTEGRATION
# ─────────────────────────────
# 7. train_model() now calls run_automl_with_agents() instead of run_automl()
#    so every training run activates the full learning AI system:
#      • RLAgent       — action selection
#      • MetaModel     — quality score + data-driven insight
#      • RetrainModel  — calibrated retrain decision
#      • AgentSystem   — workflow orchestration
#
# 8. POST /ai/feedback          → record ground-truth outcome (closes RL loop)
# 9. GET  /ai/agents/status     → full learning-state snapshot
# 10. POST /ai/agents/save      → persist all agent models to disk
#
# NaN FIX
# ─────────────────────────────
# 11. sanitize_for_json() is applied to EVERY return value that contains
#     ML result data. This replaces float nan / inf / -inf with None (JSON
#     null) and converts numpy scalars to plain Python types so Python's
#     stdlib json.dumps() never raises ValueError.
#     Applied in: train_model() final return, list_models(), get_model().
# =============================================================================

import asyncio
import math
from datetime          import datetime
from fastapi           import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic          import BaseModel
from pathlib           import Path
from typing            import Dict, Any, Optional
import joblib
import pandas as pd
import numpy  as np
from fpdf              import FPDF
from sklearn.preprocessing import PolynomialFeatures

from app.schemas.automl_schema import AutoMLRequest

# ── v5.5: agent integration wrapper ──────────────────────────────────────────
from app.services.automl_integration import (
    run_automl_with_agents,
    record_outcome,
    agent_status,
    save_agents,
    sanitize_for_json,   # ← NaN fix: deep-sanitizer for the response dict
)

# MongoDB CRUD helpers
from app.db.crud import (
    insert_model,
    get_all_models,
    get_model_by_id,
    delete_model_by_id,
    get_dataset_by_filename,
)

router = APIRouter(tags=["AI"])

MODELS_DIR = Path("models")


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: safe scalar extractor
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    """
    Convert v to float, returning None for NaN / Inf / None / non-numeric.
    Used for scalar fields stored in MongoDB (accuracy, scores, etc.)
    so the DB document itself is never written with a NaN value.
    """
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# =============================================================================
#  TRAIN  (v5.5 — uses run_automl_with_agents + sanitize_for_json)
# =============================================================================

@router.post("/train")
async def train_model(request: AutoMLRequest):
    """
    Train an AutoML model on an uploaded dataset.

    Steps
    ─────
    1. Run run_automl_with_agents() in a thread-pool executor (non-blocking).
       run_automl_with_agents() already calls sanitize_for_json() internally,
       but the dict spread below creates a new dict, so we sanitize again at
       the final return to guarantee no NaN reaches json.dumps().
    2. Resolve dataset_id from MongoDB.
    3. Build and insert model document into 'models' collection.
    4. Return the result enriched with model_id, sanitized for JSON.
    """
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_automl_with_agents,
            request.filename,
            request.target_column,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # ── 2. Resolve dataset_id ─────────────────────────────────────────────────
    dataset_id = request.dataset_id or ""
    if not dataset_id:
        try:
            ds_doc     = await get_dataset_by_filename(request.filename)
            dataset_id = ds_doc["_id"] if ds_doc else ""
        except Exception:
            dataset_id = ""

    # ── 3. Extract key scalar fields from the result ──────────────────────────
    model_name = str(
        result.get("best_model")
        or result.get("model_name")
        or result.get("model_type")
        or "Unknown"
    )
    problem_type = str(result.get("problem_type", "unknown"))

    # _safe_float guards against NaN being written to MongoDB
    raw_acc = (
        result.get("accuracy")
        or result.get("best_accuracy")
        or result.get("test_accuracy")
        or result.get("score")
    )
    accuracy = _safe_float(raw_acc)

    model_file = str(
        result.get("model_file")
        or result.get("model_path")
        or result.get("model_name")
        or ""
    )
    model_path = f"models/{model_file}" if model_file else "models/unknown.pkl"

    # Scalar metrics: float('nan') is a float so it passes isinstance checks —
    # run through _safe_float so NaN metrics are stored as None in MongoDB.
    metrics: Dict[str, Any] = {}
    skip_keys = {
        "best_model", "model_name", "model_type", "problem_type",
        "accuracy", "best_accuracy", "test_accuracy", "score",
        "model_file", "model_path",
    }
    for k, v in result.items():
        if k in skip_keys:
            continue
        if isinstance(v, (bool, str)):
            metrics[k] = v
        elif isinstance(v, (int, float)):
            # Replace NaN/Inf with None so MongoDB and JSON are both safe
            metrics[k] = _safe_float(v)
        # Skip nested dicts / lists — they go into the full result payload

    # ── 4. Insert model document into MongoDB ─────────────────────────────────
    model_doc = {
        "dataset_id":          dataset_id,
        "dataset_filename":    request.filename,
        "target_column":       request.target_column,
        "model_name":          model_name,
        "problem_type":        problem_type,
        "accuracy":            accuracy,         # None if NaN
        "model_path":          model_path,
        "created_at":          datetime.utcnow(),
        "metrics":             metrics,
        "agent_decision":      result.get("decision", ""),
        "agent_quality_score": _safe_float(result.get("model_quality_score")),
        "agent_meta_insight":  result.get("meta_insight", ""),
        "agent_retrain":       bool(result.get("retrain_decision", False)),
        "agent_workflow":      result.get("workflow_selected", ""),
        "agent_run_id":        result.get("agent_run_id", ""),
    }

    model_id: Optional[str] = None
    try:
        model_id = await insert_model(model_doc)
    except Exception as db_err:
        print(f"Warning: Failed to save model to MongoDB: {db_err}")

    # ── 5. Return sanitized response ──────────────────────────────────────────
    # sanitize_for_json() is called here even though run_automl_with_agents()
    # already sanitized `result` internally, because:
    #   a) the dict spread {**result, "model_id": ...} creates a NEW dict
    #   b) model_id (str or None) is safe, but the spread may re-introduce
    #      any numpy scalars that were in nested dicts
    # One extra sanitize call is O(n) and eliminates the ValueError for good.
    response_payload = sanitize_for_json({**result, "model_id": model_id})
    return JSONResponse(content=response_payload)


# =============================================================================
#  LIST MODELS
# =============================================================================

@router.get("/models")
async def list_models():
    """Return all model documents from MongoDB, sorted newest-first."""
    models = await get_all_models()
    return JSONResponse(content=sanitize_for_json({
        "total":  len(models),
        "models": models,
    }))


# =============================================================================
#  GET SINGLE MODEL
# =============================================================================

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Return a single model document by its MongoDB _id. 404 if not found."""
    doc = await get_model_by_id(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return JSONResponse(content=sanitize_for_json(doc))


# =============================================================================
#  DELETE MODEL
# =============================================================================

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a model: removes .pkl from disk then deletes MongoDB document.
    Returns a summary of what was removed.
    """
    doc = await get_model_by_id(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Model not found")

    removed_files = []
    pkl_path = Path(doc.get("model_path", ""))
    if pkl_path.exists():
        pkl_path.unlink()
        removed_files.append(str(pkl_path))

    deleted = await delete_model_by_id(model_id)

    return JSONResponse(content={
        "message":       "Model deleted successfully",
        "model_id":      model_id,
        "removed_files": removed_files,
        "db_deleted":    deleted,
    })


# =============================================================================
#  PREDICT REQUEST MODEL
# =============================================================================

class PredictRequest(BaseModel):
    model_name:  str
    input_data:  Dict[str, Any]


# =============================================================================
#  FEATURE ENGINEERING HELPER
# =============================================================================

def _apply_feature_engineering(df: pd.DataFrame, orig_cols: list) -> pd.DataFrame:
    """Re-applies polynomial + log + ratio engineering used in low-feature mode."""
    try:
        poly      = PolynomialFeatures(degree=2, include_bias=False)
        poly_arr  = poly.fit_transform(df[orig_cols])
        poly_cols = poly.get_feature_names_out(orig_cols)
        df_poly   = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)

        for col in df_poly.columns:
            try:
                if df_poly[col].min() > 0 and abs(float(df_poly[col].skew())) > 1.0:
                    df_poly[col] = np.log1p(df_poly[col])
            except Exception:
                pass

        for i, c1 in enumerate(orig_cols):
            for c2 in orig_cols[i + 1:]:
                rname = f"{c1}_div_{c2}"
                df_poly[rname] = (
                    df[c1] / df[c2].replace(0, np.nan)
                ).fillna(0)

        return df_poly

    except Exception as e:
        raise ValueError(f"Feature engineering failed: {e}")


# =============================================================================
#  PREDICT
# =============================================================================

@router.post("/predict")
def predict(request: PredictRequest):
    try:
        model_path = Path("models") / request.model_name
        if not model_path.exists():
            return {"error": f"Model file not found: {request.model_name}"}

        model_data    = joblib.load(model_path)
        model         = model_data["model"]
        training_cols = model_data["columns"]
        problem_type  = model_data["problem_type"]
        label_encoder = model_data.get("label_encoder", None)

        df = pd.DataFrame([request.input_data])
        df = pd.get_dummies(df)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        orig_cols = df.columns.tolist()
        if len(orig_cols) <= 3 and len(training_cols) > len(orig_cols):
            df = _apply_feature_engineering(df, orig_cols)

        df = df.reindex(columns=training_cols, fill_value=0)

        prediction = model.predict(df)
        raw_value  = prediction[0]
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()

        confidence = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba      = model.predict_proba(df)[0]
            confidence = f"{float(proba.max()) * 100:.2f}%"

        if problem_type == "classification":
            if label_encoder is not None:
                try:
                    friendly = str(label_encoder.inverse_transform([int(raw_value)])[0])
                except Exception:
                    friendly = str(raw_value)
            else:
                friendly = "Yes" if raw_value == 1 else str(raw_value)
        else:
            friendly = round(float(raw_value), 2)

        return {
            "prediction":   friendly,
            "raw_value":    raw_value,
            "confidence":   confidence,
            "problem_type": problem_type,
        }

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
#  DOWNLOAD REPORT PDF
# =============================================================================

@router.get("/download-report-pdf/{report_json}")
def download_report_pdf(report_json: str):
    import json

    report_path = Path("reports") / report_json
    if not report_path.exists():
        return {"error": "Report not found"}

    with open(report_path, "r") as f:
        report_data = json.load(f)

    pdf_name = report_json.replace(".json", ".pdf")
    pdf_path = Path("reports") / pdf_name

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Model Training Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    for key, value in report_data.items():
        if isinstance(value, (list, dict)):
            continue
        if hasattr(value, "item"):
            value = value.item()
        safe_value = str(value)[:200]
        pdf.multi_cell(0, 8, f"{key}: {safe_value}")
        pdf.ln(2)

    pdf.output(str(pdf_path))
    return FileResponse(path=pdf_path, filename=pdf_name)


# =============================================================================
#  FEEDBACK  (v5.5 — closes the RL reward loop)
# =============================================================================

class FeedbackRequest(BaseModel):
    run_id:          str
    correct:         Optional[bool]  = None
    new_cv_score:    Optional[float] = None
    retrain_helped:  Optional[bool]  = None


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    """
    Record a ground-truth outcome for a previous training run.
    Updates the RL agent's reward and the retrain model's observation history.
    """
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: record_outcome(
            run_id             = request.run_id,
            prediction_correct = request.correct,
            new_cv_score       = request.new_cv_score,
            retrain_helped     = request.retrain_helped,
        )
    )
    return JSONResponse(content=sanitize_for_json(result))


# =============================================================================
#  AGENT STATUS  (v5.5)
# =============================================================================

@router.get("/agents/status")
async def agents_status():
    """Return a complete snapshot of the learning AI system state."""
    loop   = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, agent_status)
    return JSONResponse(content=sanitize_for_json(status))


# =============================================================================
#  AGENT SAVE  (v5.5)
# =============================================================================

@router.post("/agents/save")
async def agents_save():
    """Manually persist all agent model states to disk."""
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, save_agents)
    return JSONResponse(content=sanitize_for_json(result))