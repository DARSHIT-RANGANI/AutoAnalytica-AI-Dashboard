# =============================================================================
# backend/app/modules/reports/router.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. BUG FIX: REPORT_DIR corrected to Path("app/reports")
# 2. BUG FIX: Wildcard route now checks if filename ends with .html or .pdf
#    and REDIRECTS to /static/reports/{filename} instead of trying to read
#    it as a dataset. This handles old frontend code that calls the wrong URL.
# 3. MongoDB insert after PDF generation
# 4. LIST / DETAIL / DELETE endpoints
# =============================================================================

from fastapi           import APIRouter, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pathlib           import Path
from datetime          import datetime
import pandas as pd

# MongoDB CRUD helpers
from app.db.crud import (
    insert_report,
    get_all_reports,
    get_report_by_id,
    delete_report_by_id,
    get_dataset_by_filename,
)

router = APIRouter()

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads"
REPORT_DIR = Path("app/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
#  LIST REPORTS
# =============================================================================

@router.get("/list")
async def list_reports():
    reports = await get_all_reports()
    return JSONResponse(content={
        "total":   len(reports),
        "reports": reports,
    })


# =============================================================================
#  GET SINGLE REPORT
# =============================================================================

@router.get("/detail/{report_id}")
async def get_report(report_id: str):
    doc = await get_report_by_id(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(content=doc)


# =============================================================================
#  DELETE REPORT
# =============================================================================

@router.delete("/delete/{report_id}")
async def delete_report(report_id: str):
    doc = await get_report_by_id(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")

    removed_files = []

    pdf_path = Path(doc.get("report_path", ""))
    if pdf_path.exists():
        pdf_path.unlink()
        removed_files.append(str(pdf_path))

    deleted = await delete_report_by_id(report_id)

    return JSONResponse(content={
        "message":       "Report deleted successfully",
        "report_id":     report_id,
        "removed_files": removed_files,
        "db_deleted":    deleted,
    })


# =============================================================================
#  GENERATE FULL REPORT  (wildcard — must stay LAST)
# =============================================================================

@router.get("/{filename}")
async def generate_full_report(filename: str):
    """
    ✅ FIX: If the filename is an already-generated .html or .pdf report,
    redirect straight to the static files URL so it is served correctly
    instead of being treated as a dataset to generate a report FROM.

    This handles the case where old or incorrect frontend code calls:
        GET /reports/dashboard_abc123.html
    instead of:
        GET /static/reports/dashboard_abc123.html
    """

    # ── ✅ GUARD: redirect report files to the correct static URL ─────────────
    if filename.endswith(".html") or filename.endswith(".pdf"):
        static_path = REPORT_DIR / filename
        if static_path.exists():
            # File is on disk — redirect to static server
            return RedirectResponse(
                url=f"/static/reports/{filename}",
                status_code=302,
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Report file not found on disk: {filename}. "
                       f"It may have been deleted or never generated.",
            )

    # ── Normal flow: treat filename as a dataset and generate a PDF ───────────
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        return {"error": f"File not found: {filename}"}

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as exc:
        return {"error": f"Failed to read file: {exc}"}

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf_filename = f"full_report_{filename}.pdf"
    pdf_path     = REPORT_DIR / pdf_filename

    doc      = SimpleDocTemplate(str(pdf_path))
    elements = []
    styles   = getSampleStyleSheet()

    elements.append(Paragraph("AutoAnalytica AI — Full Dataset Report", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Dataset:  {filename}",       styles["Normal"]))
    elements.append(Paragraph(f"Rows:     {df.shape[0]}",    styles["Normal"]))
    elements.append(Paragraph(f"Columns:  {df.shape[1]}",    styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Columns:", styles["Heading2"]))

    for col in df.columns:
        dtype      = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        elements.append(
            Paragraph(
                f"  • {col}  ({dtype})  —  {null_count} missing values",
                styles["Normal"],
            )
        )

    elements.append(Spacer(1, 20))
    elements.append(
        Paragraph(
            f"Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            styles["Normal"],
        )
    )

    doc.build(elements)

    # ── Resolve dataset_id ────────────────────────────────────────────────────
    dataset_id = ""
    try:
        ds_doc     = await get_dataset_by_filename(filename)
        dataset_id = ds_doc["_id"] if ds_doc else ""
    except Exception:
        pass

    # ── Insert into MongoDB ───────────────────────────────────────────────────
    report_doc = {
        "dataset_id":       dataset_id,
        "dataset_filename": filename,
        "report_filename":  pdf_filename,
        "report_path":      f"app/reports/{pdf_filename}",
        "created_at":       datetime.utcnow(),
    }

    try:
        report_id = await insert_report(report_doc)
    except Exception as db_err:
        report_id = None
        print(f"⚠️  Failed to save report to MongoDB: {db_err}")

    return JSONResponse(content={
        "report_file": pdf_filename,
        "report_id":   report_id,
        "report_url":  f"/static/reports/{pdf_filename}",
        "message":     "Report generated and saved to database.",
    })