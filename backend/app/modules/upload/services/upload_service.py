# =============================================================================
# backend/app/modules/upload/services/upload_service.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. After saving and parsing a file, one MongoDB document is inserted into
#    the 'datasets' collection via insert_dataset() from crud.py.
#
# 2. The returned dict now includes  "dataset_id"  (the new MongoDB _id as a
#    string) so the frontend can pass it to the training endpoint.
#
# 3. Human-readable file_size is calculated from the raw bytes and stored
#    in the database document.
#
# Everything else (cleaning, EDA, dashboard state) is unchanged.
# =============================================================================

import os
import math
import uuid
import pandas as pd
from pathlib  import Path
from datetime import datetime
from fastapi  import UploadFile, HTTPException

from .utils                      import generate_dashboard_data
from app.services.data_cleaner   import clean_dataframe
from app.core.state              import latest_dashboard_data
from app.services.eda_engine     import run_eda
from app.db.crud                 import insert_dataset          # ← NEW


# ── Configuration ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "uploads"


# ── Float / NaN helpers ───────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """Return float or None for nan / inf / unconvertible values."""
    try:
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _sanitize_dict(obj):
    """
    Recursively walk dicts / lists and replace nan / inf floats with None
    so FastAPI's JSONResponse never raises:
        ValueError: Out of range float values are not JSON compliant: nan
    """
    if isinstance(obj, dict):
        return {k: _sanitize_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cleaned = [_sanitize_dict(v) for v in obj]
        return tuple(cleaned) if isinstance(obj, tuple) else cleaned
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


# ── File-size helper ──────────────────────────────────────────────────────────

def _human_size(num_bytes: int) -> str:
    """Convert raw byte count to a human-readable string."""
    if num_bytes >= 1_048_576:                        # ≥ 1 MB
        return f"{num_bytes / 1_048_576:.2f} MB"
    if num_bytes >= 1_024:                            # ≥ 1 KB
        return f"{num_bytes / 1_024:.2f} KB"
    return f"{num_bytes} B"


# ── File type validation ──────────────────────────────────────────────────────

def validate_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'.  "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )
    return ext


# ── Core: save, clean, analyse, store metadata ───────────────────────────────

async def save_and_parse(file: UploadFile) -> dict:
    """
    Full pipeline:
        1. Validate extension
        2. Save raw file to UPLOAD_DIR
        3. Parse to DataFrame
        4. Auto-clean
        5. Run EDA
        6. Save cleaned file
        7. Build analysis payload
        8. *** Insert dataset metadata into MongoDB ***  ← NEW
        9. Return result dict (now includes dataset_id)
    """
    ext = validate_file_type(file.filename or "unknown")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Save raw file ──────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path   = UPLOAD_DIR / unique_name
    contents    = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}")

    # ── 3. Auto-clean ─────────────────────────────────────────────────────────
    original_rows   = df.shape[0]
    result          = clean_dataframe(df)
    df              = result["cleaned_df"]
    cleaning_summary = result["summary"]
    cleaned_rows    = df.shape[0]
    cleaning_summary["original_rows"] = int(original_rows)
    cleaning_summary["cleaned_rows"]  = int(cleaned_rows)

    # ── 4. EDA ────────────────────────────────────────────────────────────────
    eda_report = run_eda(df)

    # ── 5. Save cleaned file ──────────────────────────────────────────────────
    cleaned_filename = f"cleaned_{unique_name}"
    cleaned_path     = UPLOAD_DIR / cleaned_filename

    if ext == ".csv":
        df.to_csv(cleaned_path, index=False)
    else:
        df.to_excel(cleaned_path, index=False)

    # ── 6. Dashboard data ─────────────────────────────────────────────────────
    dashboard_data = generate_dashboard_data(df)

    # ── 7. Build analysis payload ─────────────────────────────────────────────
    rows         = int(df.shape[0])
    columns_count = int(df.shape[1])
    col_names    = df.columns.tolist()

    missing_values = {str(c): int(v) for c, v in df.isnull().sum().items()}
    duplicates     = int(df.duplicated().sum())
    column_types   = {str(c): str(t) for c, t in df.dtypes.items()}

    # Numeric summary with nan / inf guard
    numeric_summary = {}
    numeric_df      = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        stats = (
            numeric_df
            .replace([float("inf"), float("-inf")], float("nan"))
            .describe()
            .fillna(0)
        )
        for column in stats.columns:
            numeric_summary[str(column)] = {
                str(stat): _safe_float(val) or 0.0
                for stat, val in stats[column].items()
            }

    total_cells = rows * columns_count if rows > 0 and columns_count > 0 else 1

    analysis = {
        "rows":            rows,
        "columns":         columns_count,
        "missing_values":  missing_values,
        "duplicates":      duplicates,
        "column_types":    column_types,
        "numeric_summary": numeric_summary,
    }

    # ── 8. Update in-memory dashboard state (legacy) ──────────────────────────
    latest_dashboard_data.clear()
    latest_dashboard_data.update({
        "datasets_uploaded": 1,
        "models_trained":    0,
        "best_accuracy":     0,
        "recent_trainings":  [],
        "dataset_health": {
            "missing_values":  f"{round(sum(missing_values.values()) / total_cells * 100, 2)}%",
            "duplicate_rows":  f"{round((duplicates / rows * 100) if rows > 0 else 0, 2)}%",
            "class_imbalance": "Unknown",
        },
    })

    # ── 9. *** INSERT INTO MONGODB 'datasets' COLLECTION *** ──────────────────
    file_size_str = _human_size(len(contents))

    dataset_doc = {
        "filename":          unique_name,
        "original_filename": file.filename or unique_name,
        "path":              f"uploads/{unique_name}",
        "cleaned_path":      f"uploads/{cleaned_filename}",
        "columns":           col_names,
        "upload_time":       datetime.utcnow(),
        "file_size":         file_size_str,
        "row_count":         rows,
        "column_count":      columns_count,
    }

    dataset_id = await insert_dataset(dataset_doc)
    # ─────────────────────────────────────────────────────────────────────────

    # Final safety-net: sanitize every float in the response
    return _sanitize_dict({
        "dataset_id":       dataset_id,        # ← NEW: MongoDB _id string
        "original_filename": unique_name,
        "cleaned_filename":  cleaned_filename,
        "file_size":         file_size_str,
        "analysis":          analysis,
        "cleaning_summary":  cleaning_summary,
        "dashboard":         dashboard_data,
        "eda_report":        eda_report,
        "message":           "File uploaded, cleaned, analysed & saved to database.",
    })