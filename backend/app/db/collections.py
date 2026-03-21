# =============================================================================
# backend/app/db/collections.py
#
# One thin accessor function per MongoDB collection.
# Import these in crud.py (and anywhere else) instead of calling get_db()
# directly, so collection names are defined in exactly one place.
# =============================================================================

from motor.motor_asyncio import AsyncIOMotorCollection
from app.db.database import get_db


def datasets_col() -> AsyncIOMotorCollection:
    """
    'datasets' collection  →  uploaded CSV / Excel file metadata.

    Document shape:
        filename, original_filename, path, cleaned_path,
        columns, upload_time, file_size, row_count, column_count
    """
    return get_db()["datasets"]


def models_col() -> AsyncIOMotorCollection:
    """
    'models' collection  →  trained ML model artefacts (.pkl) metadata.

    Document shape:
        dataset_id, dataset_filename, target_column,
        model_name, problem_type, accuracy,
        model_path, created_at, metrics
    """
    return get_db()["models"]


def reports_col() -> AsyncIOMotorCollection:
    """
    'reports' collection  →  generated PDF report metadata.

    Document shape:
        dataset_id, dataset_filename,
        report_filename, report_path, created_at
    """
    return get_db()["reports"]