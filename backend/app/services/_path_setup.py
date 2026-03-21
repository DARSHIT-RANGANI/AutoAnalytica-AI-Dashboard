"""
_path_setup.py  —  AutoAnalytica v5.5
─────────────────────────────────────────────────────────────────────────────
Single import that resolves the `app/services/` directory and adds it to
sys.path so every sibling module (automl_service, rl_agent, meta_model, etc.)
can be imported with a plain `import automl_service` regardless of the
working directory Python was started from.

Usage  (first line of every service module)
───────────────────────────────────────────
    import _path_setup  # noqa: F401 — ensures app/services is on sys.path

Why this is needed
──────────────────
All service files live in:
    D:/autoanalytica-ai/backend/app/services/

When Python is launched from the project root (e.g. via uvicorn or pytest),
the working directory is the project root, NOT app/services/.
Python therefore cannot find `automl_service` unless the services directory
is explicitly on sys.path.

This module is the SINGLE place that handles path resolution — no other file
needs to manipulate sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Resolve the directory that contains this file (= app/services/)
# and ensure it is on sys.path so sibling modules are importable.
# ─────────────────────────────────────────────────────────────────────────────

_SERVICES_DIR = Path(__file__).resolve().parent

if str(_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICES_DIR))

# Also ensure the parent of app/services (= backend/) is on sys.path
# so `from app.services.X import ...` style imports also work.
_BACKEND_DIR = _SERVICES_DIR.parent.parent   # .../backend/
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: expose the resolved paths for debugging
# ─────────────────────────────────────────────────────────────────────────────

SERVICES_DIR: Path = _SERVICES_DIR
BACKEND_DIR:  Path = _BACKEND_DIR


def verify() -> dict:
    """
    Returns a dict showing which sibling modules are importable.
    Call from a shell to diagnose import problems:

        python -c "import _path_setup; print(_path_setup.verify())"
    """
    modules = [
        "automl_service",
        "automl_integration",
        "rl_agent",
        "meta_model",
        "retrain_model",
        "agent_system",
        "feature_extractor",
        "experience_store",
    ]
    result: dict = {"services_dir": str(SERVICES_DIR), "status": {}}
    for mod in modules:
        try:
            __import__(mod)
            result["status"][mod] = "✓ importable"
        except ImportError as exc:
            result["status"][mod] = f"✗ {exc}"
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2))