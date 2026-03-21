# =============================================================================
# backend/app/modules/dashboard/router.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. Kept the existing  GET /dashboard/{filename}  chart-generation endpoint.
#    It calls generate_dashboard_graphs() from service.py (Part 1).
#
# 2. BUG FIX: After generating the HTML dashboard file, the filename is now
#    inserted into MongoDB's 'reports' collection via insert_report().
#    Previously the HTML was saved to disk but never recorded in MongoDB,
#    so it never appeared in GET /reports/list and requests hit the wrong
#    API route (GET /reports/{filename}) instead of the static files route.
#
# 3. Added  GET /dashboard/stats   → MongoDB live stats for Dashboard.js
# 4. Added  GET /dashboard/counts  → lightweight 3-number count response
#
# ⚠️  ROUTE ORDER IS CRITICAL:
#    /stats and /counts must be defined BEFORE /{filename}.
# =============================================================================

from fastapi           import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime          import datetime
import asyncio

from app.modules.dashboard.service import (
    # Part 1 — chart engine
    generate_dashboard_graphs,
    # Part 2 — MongoDB stats
    get_dashboard_stats,
    get_summary_counts,
)

# MongoDB CRUD helpers
from app.db.crud import (
    insert_report,
    get_dataset_by_filename,
)

router = APIRouter(tags=["Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
#  SPECIFIC ROUTES FIRST  (must appear before the /{filename} wildcard)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def dashboard_stats():
    """
    Return the full MongoDB-powered stats payload for Dashboard.js.

    Response shape:
    ──────────────
    {
        "datasets_uploaded":  3,
        "models_trained":     7,
        "reports_generated":  2,
        "best_accuracy":      94.32,
        "recent_trainings":   [ { model_id, model_name, dataset, ... }, ... ]
    }
    """
    stats = await get_dashboard_stats()
    return JSONResponse(content=stats)


@router.get("/counts")
async def dashboard_counts():
    """
    Return the three entity counts only.
    Cheap endpoint for header widgets or periodic polling.

    Response shape:
    ──────────────
    { "datasets": 3, "models": 7, "reports": 2 }
    """
    counts = await get_summary_counts()
    return JSONResponse(content=counts)


# ─────────────────────────────────────────────────────────────────────────────
#  WILDCARD ROUTE LAST
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{filename}")
async def dashboard(filename: str):
    """
    Generate and return dashboard chart filenames for the given CSV/XLSX.

    Steps:
        1. Run generate_dashboard_graphs() in a thread-pool executor
        2. ✅ FIX: Insert the generated HTML filename into MongoDB reports collection
        3. Return the result (includes the HTML filename for the frontend)

    Returns:
        {
            "charts":          ["dashboard_abc123.html"],
            "rows":            1200,
            "columns":         8,
            "numeric_columns": 5,
            "cat_columns":     3,
            "missing_cells":   12,
            "detected_hue":    "status",
            "detected_hue2":   "region",
            "detected_date":   null,
            "report_id":       "64f3a..."   ← NEW: MongoDB _id of the saved report
        }
    """
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, generate_dashboard_graphs, filename)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # ── ✅ FIX: Save the generated HTML report to MongoDB reports collection ──
    # Previously the HTML was written to disk but never recorded in MongoDB.
    # This caused the Reports page to call GET /reports/{html_filename} (the
    # API route) instead of GET /static/reports/{html_filename} (static files).
    charts = result.get("charts", [])
    if charts:
        html_filename = charts[0]  # e.g. "dashboard_fa2df4af0e5f.html"

        # Optionally resolve the dataset_id from MongoDB
        dataset_id = ""
        try:
            ds_doc     = await get_dataset_by_filename(filename)
            dataset_id = ds_doc["_id"] if ds_doc else ""
        except Exception:
            pass

        report_doc = {
            "dataset_id":       dataset_id,
            "dataset_filename": filename,
            "report_filename":  html_filename,
            "report_path":      f"app/reports/{html_filename}",
            "created_at":       datetime.utcnow(),
        }

        try:
            report_id = await insert_report(report_doc)
            result["report_id"] = report_id
        except Exception as db_err:
            print(f"⚠️  Failed to save dashboard report to MongoDB: {db_err}")
            result["report_id"] = None

    return result