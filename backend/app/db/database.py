# =============================================================================
# backend/app/db/database.py
#
# Motor (async MongoDB driver) connection manager.
# connect_db() is called on FastAPI startup via lifespan.
# close_db()   is called on FastAPI shutdown via lifespan.
# get_db()     is used everywhere else to obtain the database handle.
# =============================================================================

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional

# ── Module-level singletons ──────────────────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None
_db:     Optional[AsyncIOMotorDatabase] = None

# ── Configuration (override via environment variables) ───────────────────────
MONGO_URI = os.getenv("MONGO_URI",    "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB_NAME", "autoanalytica")


# ── Lifecycle helpers (called by main.py lifespan) ───────────────────────────

async def connect_db() -> None:
    """
    Open a Motor connection pool and verify the server is reachable.
    Raises ConnectionFailure if MongoDB is not running.
    """
    global _client, _db

    _client = AsyncIOMotorClient(MONGO_URI)
    _db     = _client[DB_NAME]

    # Ping to confirm connectivity — raises immediately if server is down
    await _client.admin.command("ping")

    print(f"✅  MongoDB connected  ▶  {MONGO_URI}  /  {DB_NAME}")


async def close_db() -> None:
    """
    Gracefully close the Motor connection pool on application shutdown.
    """
    global _client
    if _client is not None:
        _client.close()
        print("🔌  MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    """
    Return the active database handle.
    Raises RuntimeError if called before connect_db() has run.
    """
    if _db is None:
        raise RuntimeError(
            "Database not initialised.  "
            "Ensure connect_db() is awaited during application startup."
        )
    return _db