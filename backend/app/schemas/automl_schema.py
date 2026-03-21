# =============================================================================
# backend/app/schemas/automl_schema.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. Added optional  dataset_id  field.
#
#    When the frontend calls  POST /ai/train , it should now pass the
#    dataset_id that was returned by  POST /upload/api/upload  so the
#    trained model document in MongoDB can be linked back to its source
#    dataset.
#
#    dataset_id is  Optional[str]  with a default of  None  so existing
#    callers that do not yet send this field continue to work without any
#    changes.
# =============================================================================

from pydantic import BaseModel
from typing   import Optional


class AutoMLRequest(BaseModel):
    filename:      str           # on-disk filename (uuid4-based, e.g. "abc123.csv")
    target_column: str           # column the model should learn to predict
    dataset_id:    Optional[str] = None  # ← NEW: MongoDB _id of the source dataset