# =============================================================================
# backend/app/modules/upload/api/router.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. GET  /upload/api/datasets         →  list all datasets from MongoDB
# 2. GET  /upload/api/datasets/{id}    →  get a single dataset by MongoDB _id
# 3. DELETE /upload/api/datasets/{id}  →  delete DB record + physical files
#    (both the raw file and the cleaned file are removed from disk)
#
# The existing  POST /upload/api/upload  and  GET /upload/api/download/{filename}
# endpoints are unchanged.
# =============================================================================

from fastapi           import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib           import Path

from app.modules.upload.services.upload_service import save_and_parse
from app.services.automl_service                import convert_to_python

# MongoDB CRUD helpers
from app.db.crud import (
    get_all_datasets,
    get_dataset_by_id,
    delete_dataset_by_id,
)

router = APIRouter(prefix="/api", tags=["Upload"])

# Physical upload directory (relative to where uvicorn is launched — backend/)
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "uploads"


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file.

    - Validates file type (csv, xlsx, xls)
    - Saves the file and a cleaned version
    - Cleans & analyses data
    - Saves metadata to MongoDB 'datasets' collection
    - Returns analysis payload **including dataset_id** for use in training
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    result = await save_and_parse(file)

    # convert_to_python replaces all nan / inf floats with None (JSON null)
    return JSONResponse(content=convert_to_python(result))


# ─────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a previously uploaded file by its on-disk filename."""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if filename.endswith(".csv"):
        media_type = "text/csv"
    else:
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    return FileResponse(path=file_path, filename=filename, media_type=media_type)


# ─────────────────────────────────────────────────────────────────────────────
#  LIST DATASETS  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/datasets")
async def list_datasets():
    """
    Return all dataset documents from MongoDB, sorted newest-first.

    Each document contains:
        _id, filename, original_filename, path, cleaned_path,
        columns, upload_time, file_size, row_count, column_count
    """
    datasets = await get_all_datasets()
    return JSONResponse(content={
        "total":    len(datasets),
        "datasets": datasets,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  GET SINGLE DATASET  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """
    Return a single dataset document by its MongoDB _id.
    Raises 404 if not found.
    """
    doc = await get_dataset_by_id(dataset_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return JSONResponse(content=doc)


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE DATASET  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset completely:
        1. Look up the document in MongoDB to get file paths
        2. Delete the raw uploaded file from disk
        3. Delete the cleaned file from disk
        4. Delete the MongoDB document

    Returns a summary of what was removed.
    """
    # ── 1. Fetch document so we know the file paths ───────────────────────────
    doc = await get_dataset_by_id(dataset_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    removed_files = []

    # ── 2. Delete raw file ────────────────────────────────────────────────────
    raw_path = UPLOAD_DIR / doc["filename"]
    if raw_path.exists():
        raw_path.unlink()
        removed_files.append(doc["filename"])

    # ── 3. Delete cleaned file ────────────────────────────────────────────────
    cleaned_filename = doc.get("cleaned_path", "").split("/")[-1]
    if cleaned_filename:
        cleaned_path = UPLOAD_DIR / cleaned_filename
        if cleaned_path.exists():
            cleaned_path.unlink()
            removed_files.append(cleaned_filename)

    # ── 4. Delete MongoDB record ──────────────────────────────────────────────
    deleted = await delete_dataset_by_id(dataset_id)

    return JSONResponse(content={
        "message":       "Dataset deleted successfully",
        "dataset_id":    dataset_id,
        "removed_files": removed_files,
        "db_deleted":    deleted,
    })