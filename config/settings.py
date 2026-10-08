"""
settings.py — the three settings the analyses read, from the environment.

Trimmed from the Justice Lens app for this repository.
"""
import os

# asyncpg connection string to a PostgreSQL database (PostGIS optional)
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

# Anomaly brake for incident corrections in database/db.py upsert_records():
# if a sync would rewrite more than this share of matched rows, and at least
# this many rows, corrections are withheld and only new rows are inserted.
UPSERT_BRAKE_MAX_CHANGED_PCT: float = float(os.environ.get("UPSERT_BRAKE_MAX_CHANGED_PCT", "10.0"))
UPSERT_BRAKE_MIN_CHANGED: int = int(os.environ.get("UPSERT_BRAKE_MIN_CHANGED", "50"))
