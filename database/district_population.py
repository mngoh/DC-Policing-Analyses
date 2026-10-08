"""District-population denominator — table DDL, tract→district aggregation, upsert.

Additive DENOMINATOR source for the accountability tool. This module is NOT wired
into app startup (``ensure_schema``): the DDL is created on demand by the manual
loader (``scripts/load_district_population.py``) so the numbers can be loaded and
inspected before anything reads them. Nothing here touches crime_index.py, the
report modules, the frontend, or the existing tables. It persists the already-
audited output of the POC (scripts/build_district_population.py, results in
docs/district-population-findings.md) into a queryable table so a later disparity
engine can join stops against a district population share.

Two tables, two grains (a deliberate schema choice — see below):
  * ``district_population``  — one row per MPD police district (7 rows, PK 1-7).
  * ``citywide_population``  — one row, the whole-city baseline the representation
    index compares each district against.

Why the citywide baseline is a SEPARATE table, not a district-0 sentinel row
----------------------------------------------------------------------------
The alternative — a sentinel row (district 0 / 'ALL') inside district_population —
was rejected. ``district`` is a real domain value 1-7 (it joins to Stop Data's
STOP_DISTRICT); a magic 0 row is a different GRAIN (whole city, not a district)
living in the detail table. Every consumer would then have to remember to exclude
it: ``SELECT SUM(total_population) FROM district_population`` would double-count
(the 7 districts *plus* the city total), and any join of stops→districts would
have to carry a ``WHERE district BETWEEN 1 AND 7`` guard forever. A separate
one-row table keeps the detail table's arithmetic correct with no filter, keeps
the two grains from mixing, and gives the baseline its own provenance. The
citywide total is computed from the FULL tract set (independent of the district
assignment step), so it is the honest DC denominator even if a tract ever fails
to land in a district.

The whole-tract centroid caveat (docs/district-population-findings.md) travels
with the data in ``source_note`` — see ``CENTROID_CAVEAT`` below.
"""
from __future__ import annotations

from typing import Any

# ── Sources (mirrors scripts/build_district_population.py, verified live) ──────
DISTRICTS_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/9"          # "Police Districts (MPD)"
)
ACS_TRACT_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Demographic_WebMercator/MapServer/38"           # "ACS 5-Year Demographic ... Census Tracts"
)
CENSUS_API = "https://api.census.gov/data/2023/acs/acs5/profile"  # DP05 profile, 2019-2023

# ACS 5-year vintage of the demographic numbers (both source paths use it).
ACS_VINTAGE = "2019-2023"

# Sanity band for the DC total (same as the POC).
DC_TOTAL_EXPECTED = (672_000, 680_000)

# The one structural weakness of this denominator, carried WITH the data so the
# methodology can never quote the numbers without the caveat (findings doc §Honest
# assessment). Every source_note ends with this sentence.
CENTROID_CAVEAT = (
    "Whole-tract centroid assignment credits each tract's full population to a "
    "single district, so population near a district boundary may be misattributed "
    "and reconciliation cannot detect it (the aggregation is complete, not accurate)."
)


def build_source_note(source_label: str) -> str:
    """Compose the ``source_note`` string: where the numbers came from + the caveat."""
    return f"{source_label} | {CENTROID_CAVEAT}"


# ── Schema (mirrors the additive style of database/stops.py) ──────────────────
# One row per MPD police district. district (1-7) is the join key to Stop Data's
# normalized STOP_DISTRICT (see database/stops.normalize_district).
DISTRICT_POPULATION_DDL = """
CREATE TABLE IF NOT EXISTS district_population (
    district         SMALLINT PRIMARY KEY CHECK (district BETWEEN 1 AND 7),
    police_district  TEXT NOT NULL,             -- '1D'-'7D' (matches Stop Data's D-suffixed form)
    total_population INTEGER NOT NULL,
    black_population INTEGER NOT NULL,
    pct_black        NUMERIC(5,2) NOT NULL,     -- Black share of the district, 0-100
    -- Provenance travels with the data (the methodology must state its origin).
    acs_vintage      TEXT NOT NULL,             -- e.g. '2019-2023'
    source_note      TEXT NOT NULL,             -- source label + the centroid caveat
    loaded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# One logical row: the whole-city baseline the representation index compares
# against. Separate table = different grain (see module docstring). scope keys it
# ('dc') so it can extend to other cities and is a clean upsert target.
CITYWIDE_POPULATION_DDL = """
CREATE TABLE IF NOT EXISTS citywide_population (
    scope            TEXT PRIMARY KEY,          -- 'dc'
    total_population INTEGER NOT NULL,
    black_population INTEGER NOT NULL,
    pct_black        NUMERIC(5,2) NOT NULL,     -- citywide Black share (the baseline, ~43.3%)
    acs_vintage      TEXT NOT NULL,
    source_note      TEXT NOT NULL,
    loaded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ── Pure aggregation (lifted verbatim from the audited POC) ───────────────────
# scripts/build_district_population.py is throwaway data-prep; this is the same
# centroid-in-polygon method it audited, moved here so it is testable without
# network/DB and so the loader and the reconciliation test share one code path.
def _point_in_rings(x: float, y: float, rings: list) -> bool:
    """Even-odd ray cast across every edge of every ring — correct for esri
    multi-ring polygons (outer rings + holes toggle the same 'inside' flag)."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
    return inside


def _centroid(rings: list) -> tuple[float, float]:
    ring = rings[0]  # outer ring; mean of vertices (matches database/ingest_census.py)
    return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))


def aggregate(tracts: list[dict], districts: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Assign each tract centroid to a district and sum. Return
    ``(district_rows, citywide, orphans)``.

    ``tracts``    : ``[{geoid, total, black, lng, lat}]``
    ``districts`` : ``[{district, police_district, rings}]`` (rings in lng/lat)

    ``district_rows`` are the seven table rows (district, police_district,
    total_population, black_population, pct_black, tracts). ``citywide`` is the
    baseline computed from the FULL tract set — independent of the assignment, so
    it stays the honest DC denominator even if a tract fails to land in a district.
    ``orphans`` are tracts whose centroid fell in no district (expected: none, as
    MPD's 7 districts tile all of DC — see the findings doc).
    """
    agg: dict[int, dict] = {
        d["district"]: {"police_district": d["police_district"], "total": 0, "black": 0, "tracts": 0}
        for d in districts
    }
    orphans: list[dict] = []
    for t in tracts:
        hit = None
        for d in districts:
            if _point_in_rings(t["lng"], t["lat"], d["rings"]):
                hit = d["district"]
                break
        if hit is None:
            orphans.append(t)
            continue
        agg[hit]["total"] += t["total"]
        agg[hit]["black"] += t["black"]
        agg[hit]["tracts"] += 1

    district_rows: list[dict] = []
    for dnum in sorted(agg):
        a = agg[dnum]
        pct = round(100 * a["black"] / a["total"], 1) if a["total"] else 0.0
        district_rows.append({
            "district": dnum,
            "police_district": a["police_district"],
            "total_population": a["total"],
            "black_population": a["black"],
            "pct_black": pct,
            "tracts": a["tracts"],
        })

    dc_total = sum(t["total"] for t in tracts)
    dc_black = sum(t["black"] for t in tracts)
    citywide = {
        "total_population": dc_total,
        "black_population": dc_black,
        "pct_black": round(100 * dc_black / dc_total, 1) if dc_total else 0.0,
    }
    return district_rows, citywide, orphans


# ── Schema + upsert ───────────────────────────────────────────────────────────
async def create_district_population_schema(conn) -> None:
    """Create the district_population + citywide_population tables (idempotent)."""
    await conn.execute(DISTRICT_POPULATION_DDL)
    await conn.execute(CITYWIDE_POPULATION_DDL)


async def upsert_district_population(conn, rows: list[dict], *, source_label: str) -> int:
    """Upsert the seven district rows (PK district). Idempotent — a re-load
    overwrites in place rather than duplicating. Returns the row count."""
    if not rows:
        return 0
    note = build_source_note(source_label)
    await conn.executemany(
        """
        INSERT INTO district_population
            (district, police_district, total_population, black_population,
             pct_black, acs_vintage, source_note, loaded_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        ON CONFLICT (district) DO UPDATE SET
            police_district  = EXCLUDED.police_district,
            total_population = EXCLUDED.total_population,
            black_population = EXCLUDED.black_population,
            pct_black        = EXCLUDED.pct_black,
            acs_vintage      = EXCLUDED.acs_vintage,
            source_note      = EXCLUDED.source_note,
            loaded_at        = NOW()
        """,
        [
            (r["district"], r["police_district"], r["total_population"],
             r["black_population"], r["pct_black"], ACS_VINTAGE, note)
            for r in rows
        ],
    )
    return len(rows)


async def upsert_citywide_population(conn, citywide: dict, *, source_label: str,
                                     scope: str = "dc") -> None:
    """Upsert the single citywide baseline row (PK scope). Idempotent."""
    note = build_source_note(source_label)
    await conn.execute(
        """
        INSERT INTO citywide_population
            (scope, total_population, black_population, pct_black,
             acs_vintage, source_note, loaded_at)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        ON CONFLICT (scope) DO UPDATE SET
            total_population = EXCLUDED.total_population,
            black_population = EXCLUDED.black_population,
            pct_black        = EXCLUDED.pct_black,
            acs_vintage      = EXCLUDED.acs_vintage,
            source_note      = EXCLUDED.source_note,
            loaded_at        = NOW()
        """,
        scope, citywide["total_population"], citywide["black_population"],
        citywide["pct_black"], ACS_VINTAGE, note,
    )
