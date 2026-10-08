"""
db.py — asyncpg connection pool, the crimes table, and the incident upsert layer.

Trimmed from the Justice Lens app for this repository: the web app's account,
newsletter and usage tables, and its alert email, are left out. The crimes
archive is written by database/ingest.py; the analysis engines read it directly.
"""
from __future__ import annotations

import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg

from config import settings as _cfg

# ── Pool ──────────────────────────────────────────────────────────────────────
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        url = _cfg.DATABASE_URL
        if not url:
            raise RuntimeError("DATABASE_URL env var not set")
        _pool = await asyncpg.create_pool(url, min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


_SCHEMA_CORE = """
CREATE TABLE IF NOT EXISTS crimes (
    id                   SERIAL PRIMARY KEY,
    ccn                  TEXT             NOT NULL,
    city                 VARCHAR(50)      NOT NULL DEFAULT 'dc',
    offense              TEXT,
    method               TEXT,
    shift                TEXT,
    report_dat           TIMESTAMPTZ,
    year                 SMALLINT,
    ward                 TEXT,
    district             TEXT,
    block                TEXT,
    neighborhood_cluster TEXT,
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    UNIQUE (ccn, city)
);

CREATE INDEX IF NOT EXISTS idx_crimes_city           ON crimes(city);
CREATE INDEX IF NOT EXISTS idx_crimes_year           ON crimes(year);
CREATE INDEX IF NOT EXISTS idx_crimes_city_year      ON crimes(city, year);
CREATE INDEX IF NOT EXISTS idx_crimes_city_year_ward ON crimes(city, year, ward);
CREATE INDEX IF NOT EXISTS idx_crimes_ward           ON crimes(ward);
CREATE INDEX IF NOT EXISTS idx_crimes_offense        ON crimes(offense);
CREATE INDEX IF NOT EXISTS idx_crimes_nbhd           ON crimes(neighborhood_cluster);
CREATE INDEX IF NOT EXISTS idx_crimes_rep_dat        ON crimes(report_dat DESC);
-- Functional index so date-cast queries can use an index scan on day-mode queries
CREATE INDEX IF NOT EXISTS idx_crimes_city_date ON crimes(city, CAST(report_dat AT TIME ZONE 'UTC' AS date));
"""

_SCHEMA_POSTGIS_COLUMN = "ALTER TABLE crimes ADD COLUMN IF NOT EXISTS location GEOMETRY(POINT, 4326);"
_SCHEMA_POSTGIS_INDEX  = "CREATE INDEX IF NOT EXISTS idx_crimes_location ON crimes USING GIST(location);"
_SCHEMA_POSTGIS_GEOG_INDEX = "CREATE INDEX IF NOT EXISTS idx_crimes_location_geog ON crimes USING GIST ((location::geography));"

# Set at runtime by ensure_schema()
_has_postgis: bool = False


async def ensure_schema() -> None:
    global _has_postgis
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Core table (no PostGIS dependency)
        await conn.execute(_SCHEMA_CORE)

        # Try to enable PostGIS — gracefully skip if not installed
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            await conn.execute(_SCHEMA_POSTGIS_COLUMN)
            # Backfill geometry for rows that predate the column (e.g. data
            # migrated from a non-PostGIS database) so the address rebuild's
            # ST_DWithin spatial join can see them.
            await conn.execute(
                "UPDATE crimes SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) "
                "WHERE location IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL"
            )
            await conn.execute(_SCHEMA_POSTGIS_INDEX)
            await conn.execute(_SCHEMA_POSTGIS_GEOG_INDEX)
            _has_postgis = True
        except Exception:
            _has_postgis = False


# ── Upsert (used by ingest.py and the daily cron) ────────────────────────────
# MPD revises records after first publication (reclassifications, corrected
# coordinates/labels). The upsert applies those corrections to existing rows
# under guards; (ccn, city) is the row identity and is NEVER updated.
_log_upsert = logging.getLogger(__name__)

# Mutable (correctable) columns, in the insert-tuple order after ccn/city.
_MUTABLE_FIELDS = (
    "offense", "method", "shift", "report_dat", "year",
    "ward", "district", "block", "neighborhood_cluster",
    "latitude", "longitude",
)
# Text columns where an incoming empty string means "no data", same as NULL.
_TEXT_FIELDS = frozenset(
    ("offense", "method", "shift", "ward", "district", "block", "neighborhood_cluster")
)

# Per-sync cap on field-by-field correction log lines, so a deliberate bulk
# rewrite collapses to a summary instead of flooding the logs.
_CORRECTION_LOG_DETAIL_CAP = 50

_INSERT_POSTGIS = """
INSERT INTO crimes (
    ccn, city, offense, method, shift, report_dat, year,
    ward, district, block, neighborhood_cluster,
    latitude, longitude, location
) VALUES (
    $1, $2, $3, $4, $5, $6, $7,
    $8, $9, $10, $11,
    $12, $13,
    CASE WHEN $12::double precision IS NOT NULL AND $13::double precision IS NOT NULL
         THEN ST_SetSRID(ST_MakePoint($13::double precision, $12::double precision), 4326)
         ELSE NULL END
)
"""

_INSERT_PLAIN = """
INSERT INTO crimes (
    ccn, city, offense, method, shift, report_dat, year,
    ward, district, block, neighborhood_cluster,
    latitude, longitude
) VALUES (
    $1, $2, $3, $4, $5, $6, $7,
    $8, $9, $10, $11,
    $12, $13
)
"""

# Coordinates only move as a complete pair — a half-missing pair keeps both
# stored values (and the stored geometry).
_COORD_PAIR_PRESENT = "EXCLUDED.latitude IS NOT NULL AND EXCLUDED.longitude IS NOT NULL"


def _guarded_update_clause(with_location: bool) -> str:
    """Render the ON CONFLICT DO UPDATE clause for the guarded upsert.

    Generated (rather than two hand-written variants) so the guard rule is
    encoded exactly once: an incoming NULL — or empty string for text fields —
    never overwrites a populated stored value, and the SET list and the
    no-op-suppressing WHERE clause can never drift apart. ccn/city are absent
    from _MUTABLE_FIELDS, so identity is structurally impossible to update.
    """
    effective = {}
    for f in _MUTABLE_FIELDS:
        if f in ("latitude", "longitude"):
            effective[f] = (
                f"CASE WHEN {_COORD_PAIR_PRESENT} THEN EXCLUDED.{f} ELSE crimes.{f} END"
            )
        elif f in _TEXT_FIELDS:
            effective[f] = f"COALESCE(NULLIF(EXCLUDED.{f}, ''), crimes.{f})"
        else:
            effective[f] = f"COALESCE(EXCLUDED.{f}, crimes.{f})"

    sets = [f"{f} = {effective[f]}" for f in _MUTABLE_FIELDS]
    if with_location:
        sets.append(
            f"location = CASE WHEN {_COORD_PAIR_PRESENT} THEN EXCLUDED.location ELSE crimes.location END"
        )

    # Only rewrite rows whose effective values actually differ — the daily
    # 14-day re-fetch must not churn thousands of identical rows (dead
    # tuples/WAL) every run.
    effective_tuple = ",\n       ".join(effective[f] for f in _MUTABLE_FIELDS)
    stored_tuple = ", ".join(f"crimes.{f}" for f in _MUTABLE_FIELDS)
    return (
        "ON CONFLICT (ccn, city) DO UPDATE SET\n    "
        + ",\n    ".join(sets)
        + f"\nWHERE ({effective_tuple})\n"
        + f"      IS DISTINCT FROM ({stored_tuple})\n"
    )


# Guarded upsert: inserts new rows, applies MPD corrections to existing ones.
_UPSERT_SQL_POSTGIS = _INSERT_POSTGIS + _guarded_update_clause(with_location=True)
_UPSERT_SQL_PLAIN   = _INSERT_PLAIN   + _guarded_update_clause(with_location=False)

# Legacy insert-only variants — the anomaly brake falls back to these so a
# degraded feed can still add new records but cannot touch stored ones.
_ON_CONFLICT_IGNORE = "ON CONFLICT (ccn, city) DO NOTHING\n"
_INSERT_ONLY_SQL_POSTGIS = _INSERT_POSTGIS + _ON_CONFLICT_IGNORE
_INSERT_ONLY_SQL_PLAIN   = _INSERT_PLAIN   + _ON_CONFLICT_IGNORE

_PREREAD_SQL = (
    "SELECT ccn, " + ", ".join(_MUTABLE_FIELDS)
    + " FROM crimes WHERE city = $1 AND ccn = ANY($2::text[])"
)


def _epoch_ms_to_dt(epoch_ms) -> Optional[datetime]:
    """Convert ArcGIS epoch-milliseconds integer to timezone-aware datetime."""
    if epoch_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(epoch_ms) / 1000, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _arcgis_record_to_row(r: dict, city: str = "dc") -> Optional[tuple]:
    """
    Convert a raw ArcGIS attributes dict to a tuple for the upsert query.
    Returns None if CCN is missing (can't deduplicate without it).
    """
    ccn = (r.get("CCN") or "").strip()
    if not ccn:
        return None

    # `year` is derived solely from `report_dat` so the two can never diverge. If
    # the timestamp is missing/unparseable, year is left NULL too, and the guarded
    # upsert's NULL-guard then protects both together. The previous string-prefix
    # fallback could set `year` without a `report_dat`, which on a correction let
    # `year` move across a boundary while the stored timestamp stayed put — a row
    # reading year=2024 but report_dat in 2023. ArcGIS `f=json` always emits
    # REPORT_DAT as epoch-ms, so nothing real depended on that fallback.
    report_dt = _epoch_ms_to_dt(r.get("REPORT_DAT"))
    year: Optional[int] = report_dt.year if report_dt else None

    lat = r.get("LATITUDE")
    lng = r.get("LONGITUDE")
    try:
        lat = float(lat) if lat is not None else None
        lng = float(lng) if lng is not None else None
    except (ValueError, TypeError):
        lat = lng = None

    return (
        ccn,
        city,
        (r.get("OFFENSE") or "").upper() or None,
        r.get("METHOD"),
        r.get("SHIFT"),
        report_dt,
        year,
        r.get("WARD"),
        r.get("DISTRICT"),
        r.get("BLOCK"),
        r.get("NEIGHBORHOOD_CLUSTER"),
        lat,
        lng,
    )


def _effective_update(stored: dict, incoming: dict) -> dict:
    """Python mirror of the SQL guard: the values the DO UPDATE would leave in
    place for one matched row. Incoming NULL (or empty string for text fields)
    never overwrites a populated stored value; coordinates only move as a
    complete pair. Used by the anomaly brake and the correction change-log —
    the SQL enforces the same rule independently at write time."""
    out = {}
    for f in _MUTABLE_FIELDS:
        if f in ("latitude", "longitude"):
            continue
        new = incoming.get(f)
        if f in _TEXT_FIELDS and new == "":
            new = None
        out[f] = stored[f] if new is None else new
    if incoming.get("latitude") is not None and incoming.get("longitude") is not None:
        out["latitude"], out["longitude"] = incoming["latitude"], incoming["longitude"]
    else:
        out["latitude"], out["longitude"] = stored["latitude"], stored["longitude"]
    return out


def _row_as_incoming(row: tuple) -> dict:
    """Map an insert-tuple from _arcgis_record_to_row back to named fields."""
    return dict(zip(("ccn", "city") + _MUTABLE_FIELDS, row))


async def _compute_corrections(conn, rows: list[tuple], city: str):
    """Pre-read the already-stored rows this sync matches and compute which
    would actually change under the guard semantics.

    Returns (matched_count, corrections) where corrections maps
    ccn -> [(field, old, new), …] for rows with at least one effective change.
    """
    incoming_by_ccn = {}
    for r in rows:
        # Last occurrence wins, matching executemany's sequential outcome.
        incoming_by_ccn[r[0]] = _row_as_incoming(r)

    existing = await conn.fetch(_PREREAD_SQL, city, list(incoming_by_ccn.keys()))

    corrections: dict[str, list[tuple]] = {}
    for stored_row in existing:
        stored = dict(stored_row)
        eff = _effective_update(stored, incoming_by_ccn[stored["ccn"]])
        diffs = [(f, stored[f], eff[f]) for f in _MUTABLE_FIELDS if eff[f] != stored[f]]
        if diffs:
            corrections[stored["ccn"]] = diffs
    return len(existing), corrections


def _log_corrections(corrections: dict, city: str) -> None:
    for i, (ccn, diffs) in enumerate(corrections.items()):
        if i >= _CORRECTION_LOG_DETAIL_CAP:
            _log_upsert.info(
                "[upsert] …and %d more corrected rows (detail capped at %d)",
                len(corrections) - _CORRECTION_LOG_DETAIL_CAP, _CORRECTION_LOG_DETAIL_CAP,
            )
            break
        for field, old, new in diffs:
            _log_upsert.info(
                "[upsert] correction city=%s ccn=%s %s: %r -> %r", city, ccn, field, old, new
            )


async def upsert_records(
    records: list[dict], city: str = "dc", *, allow_bulk_rewrite: bool = False
) -> int:
    """
    Bulk-upsert a list of ArcGIS attribute dicts for a given city.

    New (ccn, city) rows insert as before. Already-stored rows now receive MPD
    corrections via a guarded DO UPDATE:
      - identity (ccn, city) is never modified;
      - an incoming NULL/empty value never overwrites a populated stored value
        (a degraded ArcGIS response cannot blank good data);
      - coordinates only update as a complete lat/lng pair;
      - anomaly brake: if the sync would rewrite > UPSERT_BRAKE_MAX_CHANGED_PCT
        of the matched rows AND at least UPSERT_BRAKE_MIN_CHANGED rows, the
        corrections are NOT applied — a sample is logged and the sync falls back to insert-only (the exact
        pre-guarded-upsert behavior).

    ``allow_bulk_rewrite=True`` suppresses the brake for a deliberate backfill
    (``python database/ingest.py --allow-bulk-rewrite``). It never disables the
    NULL/empty or identity guards. Default OFF; the daily cron never sets it.

    Returns the attempted-row count (unchanged contract).
    """
    pool = await get_pool()
    rows = [_arcgis_record_to_row(r, city) for r in records]
    rows = [r for r in rows if r is not None]
    if not rows:
        return 0

    sql = _UPSERT_SQL_POSTGIS if _has_postgis else _UPSERT_SQL_PLAIN
    async with pool.acquire() as conn:
        matched, corrections = await _compute_corrections(conn, rows, city)
        changed = len(corrections)
        changed_pct = (changed / matched * 100) if matched else 0.0

        brake_tripped = (
            not allow_bulk_rewrite
            and changed >= _cfg.UPSERT_BRAKE_MIN_CHANGED
            and changed_pct > _cfg.UPSERT_BRAKE_MAX_CHANGED_PCT
        )
        if brake_tripped:
            _log_upsert.warning(
                "[upsert] anomaly brake tripped for city=%s: %d of %d matched rows would "
                "change (%.1f%% > %.1f%%, >= %d) — corrections NOT applied, inserting new "
                "rows only. Re-run with --allow-bulk-rewrite if intentional.",
                city, changed, matched, changed_pct,
                _cfg.UPSERT_BRAKE_MAX_CHANGED_PCT, _cfg.UPSERT_BRAKE_MIN_CHANGED,
            )
            sample = [
                f"CCN {ccn} {field}: {old!r} -> {new!r}"
                for ccn, diffs in list(corrections.items())[:5]
                for field, old, new in diffs[:1]
            ]
            _log_upsert.warning("[upsert] sample of the withheld corrections: %s", "; ".join(sample))
            sql = _INSERT_ONLY_SQL_POSTGIS if _has_postgis else _INSERT_ONLY_SQL_PLAIN
        elif corrections:
            _log_corrections(corrections, city)

        await conn.executemany(sql, rows)

    return len(rows)
