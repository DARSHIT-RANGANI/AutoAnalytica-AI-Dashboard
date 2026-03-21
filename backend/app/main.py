# =============================================================================
# backend/app/main.py
#
# CHANGES vs previous version
# ─────────────────────────────
# 1. MongoDB lifespan  — connect_db() on startup, close_db() on shutdown
#    Uses the modern  @asynccontextmanager  lifespan pattern (not deprecated
#    on_event decorators).
#
# 2. Static reports mount moved from  /reports  →  /static/reports
#    The old  /reports  prefix caused a routing conflict: FastAPI's main
#    router intercepted ALL  GET /reports/*  requests before the static
#    sub-app could serve the PDF files.
#    Serving PDFs:   GET /static/reports/{filename}.pdf
#    API endpoints:  GET /reports/list   DELETE /reports/delete/{id}   etc.
#
# 3. Version bumped to 0.2.1
#
# FIX (v0.2.1)
# ─────────────────────────────
# Removed the duplicate  POST /ai/predict  endpoint that was defined here
# AND inside  app/modules/ai/router.py  (@router.post("/predict") mounted
# at prefix "/ai").  FastAPI silently served only the first registration;
# the router version is the correct one — it handles label_encoder,
# feature engineering, and full column alignment.
#
# Removed:   class PredictRequest (BaseModel)
#            @app.post("/ai/predict")  async def predict(...)
#
# Kept:      def _sanitize(obj)  — utility used elsewhere in the codebase.
# =============================================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import math
from typing import Any, Dict

from app.modules.reports.router    import router as reports_router
from app.modules.upload.api.router import router as upload_router
from app.modules.ai.router         import router as ai_router
from app.modules.dashboard.router  import router as dashboard_router

# MongoDB helpers
from app.db.database import connect_db, close_db


# ── Application lifespan (startup + shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs connect_db() before the first request is handled and
    close_db() after the last one — even on unexpected shutdown.
    """
    await connect_db()   # ← open Motor connection pool
    yield                # ← application is running
    await close_db()     # ← close connection pool gracefully


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoAnalytica AI",
    description="Production-ready AI Data Analyst SaaS Backend",
    version="0.2.1",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ───────────────────────────────────────────────────────
# Reports PDFs are served at:  GET /static/reports/{filename}.pdf
# Physical location:           backend/app/reports/
#
# The path  /static/reports  avoids the conflict with the  /reports  API router.
# Update any frontend download URL from  /reports/{file}  →  /static/reports/{file}
app.mount(
    "/static/reports",
    StaticFiles(directory="app/reports"),
    name="static_reports",
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(reports_router,   prefix="/reports",   tags=["Reports"])
app.include_router(upload_router,    prefix="/upload",    tags=["Upload"])
app.include_router(ai_router,        prefix="/ai",        tags=["AI"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    return {
        "status":  "ok",
        "service": "AutoAnalytica AI",
        "version": "0.2.1",
    }


@app.get("/", tags=["System"])
async def root():
    return {"message": "AutoAnalytica AI Backend Running 🚀"}


# ── Utility ───────────────────────────────────────────────────────────────────
def _sanitize(obj: Any) -> Any:
    """Recursively replace nan / inf with None so JSONResponse never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [_sanitize(v) for v in obj]
        return tuple(sanitized) if isinstance(obj, tuple) else sanitized
    return obj