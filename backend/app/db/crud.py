# =============================================================================
# backend/app/db/crud.py
#
# All async CRUD operations for the three MongoDB collections:
#   - datasets
#   - models
#   - reports
#
# Rules:
#   • Every function that returns a document calls _str_id() so the
#     returned dict is always JSON-serialisable (no raw ObjectId objects).
#   • _to_oid() validates the incoming id string and raises HTTP 400 on
#     bad input instead of letting Bson throw an unhandled exception.
#   • delete_* functions also accept the caller's responsibility for
#     physically removing files — they only delete the DB record.
# =============================================================================

from bson        import ObjectId
from bson.errors import InvalidId
from datetime    import datetime
from typing      import Optional, List

from fastapi     import HTTPException

from app.db.collections import datasets_col, models_col, reports_col


# ── Shared helpers ────────────────────────────────────────────────────────────

def _str_id(doc: Optional[dict]) -> Optional[dict]:
    """
    Convert the MongoDB ObjectId in '_id' to a plain string so the
    document can be returned directly in a JSONResponse.
    Returns None if the document itself is None.
    """
    if doc is None:
        return None
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def _to_oid(id_str: str) -> ObjectId:
    """
    Parse a string → ObjectId.
    Raises HTTP 400 (Bad Request) with a clear message on invalid input
    instead of letting Bson raise an unhandled InvalidId error.
    """
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"'{id_str}' is not a valid MongoDB ObjectId.",
        )


# =============================================================================
#  DATASETS
# =============================================================================

async def insert_dataset(data: dict) -> str:
    """
    Insert a dataset document.
    Returns the new document's _id as a string.
    """
    result = await datasets_col().insert_one(data)
    return str(result.inserted_id)


async def get_all_datasets() -> List[dict]:
    """
    Return all dataset documents sorted newest-first.
    """
    cursor = datasets_col().find().sort("upload_time", -1)
    return [_str_id(doc) async for doc in cursor]


async def get_dataset_by_id(dataset_id: str) -> Optional[dict]:
    """
    Find a dataset by its MongoDB _id string.
    Returns the document or None if not found.
    """
    doc = await datasets_col().find_one({"_id": _to_oid(dataset_id)})
    return _str_id(doc)


async def get_dataset_by_filename(filename: str) -> Optional[dict]:
    """
    Find a dataset by its on-disk filename (the uuid4-based name).
    Useful for looking up dataset_id during model training.
    """
    doc = await datasets_col().find_one({"filename": filename})
    return _str_id(doc)


async def delete_dataset_by_id(dataset_id: str) -> bool:
    """
    Delete a dataset record from MongoDB.
    Returns True if a document was deleted, False if not found.
    NOTE: The caller is responsible for deleting the physical file.
    """
    result = await datasets_col().delete_one({"_id": _to_oid(dataset_id)})
    return result.deleted_count > 0


async def count_datasets() -> int:
    """Return the total number of dataset documents."""
    return await datasets_col().count_documents({})


# =============================================================================
#  MODELS
# =============================================================================

async def insert_model(data: dict) -> str:
    """
    Insert a model document.
    Returns the new document's _id as a string.
    """
    result = await models_col().insert_one(data)
    return str(result.inserted_id)


async def get_all_models() -> List[dict]:
    """
    Return all model documents sorted newest-first.
    """
    cursor = models_col().find().sort("created_at", -1)
    return [_str_id(doc) async for doc in cursor]


async def get_model_by_id(model_id: str) -> Optional[dict]:
    """
    Find a model by its MongoDB _id string.
    Returns the document or None if not found.
    """
    doc = await models_col().find_one({"_id": _to_oid(model_id)})
    return _str_id(doc)


async def delete_model_by_id(model_id: str) -> bool:
    """
    Delete a model record from MongoDB.
    Returns True if a document was deleted, False if not found.
    NOTE: The caller is responsible for deleting the physical .pkl file.
    """
    result = await models_col().delete_one({"_id": _to_oid(model_id)})
    return result.deleted_count > 0


async def count_models() -> int:
    """Return the total number of model documents."""
    return await models_col().count_documents({})


async def get_best_accuracy() -> float:
    """
    Return the highest accuracy (0–100 %) across all trained models.
    Returns 0.0 if no models exist.
    """
    pipeline = [
        {"$match":  {"accuracy": {"$ne": None}}},
        {"$group":  {"_id": None, "max_acc": {"$max": "$accuracy"}}},
    ]
    async for doc in models_col().aggregate(pipeline):
        val = doc.get("max_acc")
        return round(float(val) * 100, 2) if val is not None else 0.0
    return 0.0


async def get_recent_models(limit: int = 5) -> List[dict]:
    """
    Return the *limit* most recently trained model documents.
    Used by the dashboard to show recent-training activity.
    """
    cursor = models_col().find().sort("created_at", -1).limit(limit)
    return [_str_id(doc) async for doc in cursor]


# =============================================================================
#  REPORTS
# =============================================================================

async def insert_report(data: dict) -> str:
    """
    Insert a report document.
    Returns the new document's _id as a string.
    """
    result = await reports_col().insert_one(data)
    return str(result.inserted_id)


async def get_all_reports() -> List[dict]:
    """
    Return all report documents sorted newest-first.
    """
    cursor = reports_col().find().sort("created_at", -1)
    return [_str_id(doc) async for doc in cursor]


async def get_report_by_id(report_id: str) -> Optional[dict]:
    """
    Find a report by its MongoDB _id string.
    Returns the document or None if not found.
    """
    doc = await reports_col().find_one({"_id": _to_oid(report_id)})
    return _str_id(doc)


async def delete_report_by_id(report_id: str) -> bool:
    """
    Delete a report record from MongoDB.
    Returns True if a document was deleted, False if not found.
    NOTE: The caller is responsible for deleting the physical PDF file.
    """
    result = await reports_col().delete_one({"_id": _to_oid(report_id)})
    return result.deleted_count > 0


async def count_reports() -> int:
    """Return the total number of report documents."""
    return await reports_col().count_documents({})