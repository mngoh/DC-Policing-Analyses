"""Moving-violation counts (camera vs officer) — DDL, district assignment, engine.

The camera-vs-officer analysis: where MPD officers choose to write traffic
tickets versus where automated cameras record violations. Tickets carry no
race, so this is a GEOGRAPHY-OF-DISCRETION analysis, not a racial-disparity
index; the district Black-population share rides beside the result as context,
never as a denominator.

Source: Open Data DC monthly "Moving Violations" tables (one layer per month:
``Violations_Moving_{year}/MapServer/{month-1}``, ids 0-11 = Jan-Dec; verified
live 2026-07-13). The layers are TABLES — no geometry, so ArcGIS spatial
filters silently return unfiltered counts (verified: a polygon filter changed
nothing). District assignment is therefore done HERE, by point-in-polygon
against the real MPD district polygons (Public_Safety layer 9), using each
ticket's LATITUDE/LONGITUDE:
  * CAMERA tickets (ISSUING_AGENCY_CODE 92, "Regional Operations Command-
    North" — verified: its violations are exclusively automated-enforcement
    types) come from FIXED sites: ~350 distinct LOCATION strings per month.
    One server-side GROUP BY LOCATION per month with avg lat/lng, then each
    site is assigned to a district once.
  * OFFICER tickets are MPD agency codes 1-7 (the seven districts) plus 25
    (Special Operations/Traffic Division). Low volume (~3-4k/month) — fetched
    as raw lat/lng rows and assigned per ticket. Assignment is by TICKET
    LOCATION, not the issuing officer's district assignment.
  * Everything else (federal agencies, taxi commission, housing authority,
    'NO AGENCY') is excluded from the comparison and counted.

Storage grain: (month, source, district) counts — the finest grain that is
practical (raw camera rows are ~260k/month and carry nothing the analysis
needs beyond location). Batch-versioned like stops/arrests; "current" counts
are the latest FINISHED batch.
"""
from __future__ import annotations

from typing import Any, Optional

DISTRICT_POLYGONS_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/9"
)

CAMERA_AGENCY_CODE = 92                    # Regional Operations Command-North
OFFICER_AGENCY_CODES = (1, 2, 3, 4, 5, 6, 7, 25)   # district officers + SOD/Traffic

SOURCES = ("camera", "officer")


def month_layer_url(year: int, month: int) -> str:
    """One table per month: id 0 = January ... 11 = December (verified live)."""
    return (f"https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
            f"Violations_Moving_{year}/MapServer/{month - 1}")


# ── Point-in-polygon (pure; even-odd ray casting over all rings) ──────────────
def point_in_rings(lng: float, lat: float, rings: list[list[list[float]]]) -> bool:
    """Even-odd test in lon/lat space. Rings are ArcGIS polygon rings
    ([[x, y], ...]); holes fall out naturally under even-odd counting."""
    inside = False
    for ring in rings:
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if (yi > lat) != (yj > lat):
                x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
                if lng < x_cross:
                    inside = not inside
            j = i
    return inside


def assign_district(lng: Optional[float], lat: Optional[float],
                    polygons: dict[int, list]) -> Optional[int]:
    """District 1-7 for a point, or None (null coords / outside every polygon)."""
    if lng is None or lat is None:
        return None
    for d, rings in polygons.items():
        if point_in_rings(lng, lat, rings):
            return d
    return None


# ── Schema ────────────────────────────────────────────────────────────────────
TICKET_BATCHES_DDL = """
CREATE TABLE IF NOT EXISTS ticket_count_batches (
    batch_id      BIGSERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    months        INTEGER,
    note          TEXT
);
"""

TICKET_COUNTS_DDL = """
CREATE TABLE IF NOT EXISTS moving_violation_counts (
    id            SERIAL PRIMARY KEY,
    ingest_batch  BIGINT NOT NULL REFERENCES ticket_count_batches(batch_id),
    month         DATE NOT NULL,          -- first of month
    source        TEXT NOT NULL,          -- camera | officer | other_agency
    district      SMALLINT,               -- 1-7; NULL = unassigned/inapplicable
    n             INTEGER NOT NULL,
    UNIQUE (ingest_batch, month, source, district)
);
CREATE INDEX IF NOT EXISTS idx_mvc_batch ON moving_violation_counts(ingest_batch);
"""


async def create_tickets_schema(conn) -> None:
    await conn.execute(TICKET_BATCHES_DDL)
    await conn.execute(TICKET_COUNTS_DDL)


async def start_batch(conn, note: Optional[str] = None) -> int:
    return await conn.fetchval(
        "INSERT INTO ticket_count_batches (note) VALUES ($1) RETURNING batch_id", note)


async def finish_batch(conn, batch_id: int, months: int) -> None:
    await conn.execute(
        "UPDATE ticket_count_batches SET finished_at=NOW(), months=$2 WHERE batch_id=$1",
        batch_id, months)


async def upsert_counts(conn, rows: list[tuple]) -> int:
    """rows: (ingest_batch, month, source, district, n)."""
    if not rows:
        return 0
    await conn.executemany(
        "INSERT INTO moving_violation_counts (ingest_batch, month, source, district, n) "
        "VALUES ($1, $2, $3, $4, $5) "
        "ON CONFLICT (ingest_batch, month, source, district) DO UPDATE SET n=EXCLUDED.n",
        rows)
    return len(rows)


# ── Engine ────────────────────────────────────────────────────────────────────
ENFORCEMENT_DEFINITION = (
    "For each district: officer share = that district's MPD officer-written "
    "tickets / all district-assigned officer tickets; camera share = the same "
    "for automated-camera tickets; skew = officer share / camera share. "
    "Both sides are assigned by TICKET LOCATION (point-in-polygon against MPD "
    "district polygons). Officer = MPD agency codes 1-7 and 25 (SOD/Traffic); "
    "camera = code 92 (Regional Operations Command-North, verified automated). "
    "Other agencies and unassignable locations are excluded and reported. "
    "Tickets carry no race: this is a geography-of-discretion comparison, and "
    "camera placement is itself a policy choice, not a neutral measure of "
    "violations."
)


def skew_from_counts(rows: list[dict], district_meta: Optional[dict] = None) -> dict:
    """Pure aggregation: (month, source, district, n) rows -> per-district skew.

    Excluded categories (source 'other_agency', NULL districts) are summed and
    reported, never silently dropped.
    """
    district_meta = district_meta or {}
    officer: dict[int, int] = {}
    camera: dict[int, int] = {}
    excluded = {"other_agency": 0, "camera_unassigned": 0, "officer_unassigned": 0}
    months = set()
    for r in rows:
        months.add(r["month"])
        d, s, n = r.get("district"), r["source"], r["n"]
        if s == "other_agency":
            excluded["other_agency"] += n
        elif d is None:
            excluded[f"{s}_unassigned"] += n
        elif s == "officer":
            officer[d] = officer.get(d, 0) + n
        elif s == "camera":
            camera[d] = camera.get(d, 0) + n
    officer_total, camera_total = sum(officer.values()), sum(camera.values())

    districts = []
    for d in sorted(set(officer) | set(camera)):
        o_share = officer.get(d, 0) / officer_total if officer_total else None
        c_share = camera.get(d, 0) / camera_total if camera_total else None
        districts.append({
            "district": d,
            "officer_tickets": officer.get(d, 0),
            "camera_tickets": camera.get(d, 0),
            "officer_share": o_share,
            "camera_share": c_share,
            "skew": (o_share / c_share) if o_share is not None and c_share else None,
            "black_pop_share": district_meta.get(d, {}).get("black_pop_share"),
            "police_district": district_meta.get(d, {}).get("police_district"),
        })
    return {
        "districts": districts,
        "officer_total": officer_total,
        "camera_total": camera_total,
        "months": sorted(months),
        "excluded": excluded,
        "definition": ENFORCEMENT_DEFINITION,
    }


_LATEST_TICKET_BATCH_SQL = """
SELECT MAX(batch_id) FROM ticket_count_batches WHERE finished_at IS NOT NULL
"""


async def enforcement_skew(conn, *, batch_id: Optional[int] = None) -> dict:
    """Camera-vs-officer skew per district, latest finished batch.

    Raises RuntimeError both when the batch is missing AND when the tables do
    not exist yet (first deploy before any ingest) so callers keep one
    graceful-degradation contract.
    """
    import asyncpg
    from database.disparity import _fetch_district_meta
    try:
        batch = batch_id if batch_id is not None else await conn.fetchval(_LATEST_TICKET_BATCH_SQL)
    except asyncpg.UndefinedTableError:
        raise RuntimeError(
            "ticket-count tables not created yet - run scripts/ingest_ticket_counts.py first.")
    if batch is None:
        raise RuntimeError(
            "no finished ticket-count batch — run scripts/ingest_ticket_counts.py first.")
    rows = await conn.fetch(
        "SELECT month, source, district, n FROM moving_violation_counts "
        "WHERE ingest_batch = $1", batch)
    district_meta = await _fetch_district_meta(conn)
    result = skew_from_counts([dict(r) for r in rows], district_meta)
    result["meta"] = {"ingest_batch": batch, "grain": "ticket counts by month/source/district"}
    return result
