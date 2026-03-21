# =============================================================================
# backend/app/db/schemas.py
#
# Pydantic models that describe the shape of every MongoDB document.
# These are NOT FastAPI request/response models — they are used internally
# to validate data before inserting it into MongoDB.
#
# Usage:
#   doc = DatasetDocument(**data).model_dump()
#   dataset_id = await insert_dataset(doc)
# =============================================================================

from pydantic import BaseModel, Field
from typing  import Optional, List, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
#  datasets  collection
# ─────────────────────────────────────────────────────────────────────────────

class DatasetDocument(BaseModel):
    """
    Metadata stored for every uploaded dataset.

    Path convention:  uploads/{filename}
    """
    filename:           str            # unique on-disk name  (uuid4 + ext)
    original_filename:  str            # name the user uploaded
    path:               str            # relative path  →  uploads/{filename}
    cleaned_path:       str            # relative path  →  uploads/cleaned_{filename}
    columns:            List[str]      # column headers after cleaning
    upload_time:        datetime = Field(default_factory=datetime.utcnow)
    file_size:          str            # human-readable  e.g. "2.30 MB"
    row_count:          int
    column_count:       int


# ─────────────────────────────────────────────────────────────────────────────
#  models  collection
# ─────────────────────────────────────────────────────────────────────────────

class ModelDocument(BaseModel):
    """
    Metadata stored for every trained ML model artefact.

    Path convention:  models/{model_filename}.pkl
    """
    dataset_id:       str                    # str(ObjectId) of parent dataset
    dataset_filename: str                    # filename on disk (not original)
    target_column:    str                    # column the model was trained to predict
    model_name:       str                    # e.g. "RandomForest", "XGBoost"
    problem_type:     str                    # "classification"  |  "regression"
    accuracy:         Optional[float] = None # 0.0 – 1.0  (None if unavailable)
    model_path:       str                    # relative  →  models/{file}.pkl
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    metrics:          Optional[Dict[str, Any]] = None   # full metrics dict from automl


# ─────────────────────────────────────────────────────────────────────────────
#  reports  collection
# ─────────────────────────────────────────────────────────────────────────────

class ReportDocument(BaseModel):
    """
    Metadata stored for every generated PDF report.

    Path convention:  app/reports/{report_filename}.pdf
    IMPORTANT: physical files live at  backend/app/reports/
    """
    dataset_id:       Optional[str] = None   # str(ObjectId) — may be unknown
    dataset_filename: str                    # source dataset filename
    report_filename:  str                    # file name only  e.g. full_report_sales.pdf
    report_path:      str                    # relative  →  app/reports/{filename}
    created_at:       datetime = Field(default_factory=datetime.utcnow)