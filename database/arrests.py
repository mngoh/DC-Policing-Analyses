"""MPD Adult Arrests — table DDL, batch versioning, and upsert.

Mirrors database/stops.py exactly: manual, additive, snapshot-per-release batch
versioning; NOT wired into app startup. The DDL is created on demand by
scripts/ingest_arrests.py. Nothing here touches the stops tables, crime_index,
or the frontend.

Source: "Adult Arrests", DC ArcGIS Public_Safety_WebMercator/MapServer/38
(2013-2025, ~320k rows; verified live 2026-07-12). One row per arrest.

Value shapes (verified live):
  * RACE: BLACK / WHITE / ASIAN / OTHER / MULTIPLE plus TWO unknown labels
    ('UNK' and 'UNKNOWN') — stored verbatim; the engine treats both as
    not-known-race, same contract as the stops engine's 'Unknown'.
  * ETHNICITY here is a separate HISPANIC flag (NOT HISPANIC / HISPANIC /
    UNKNOWN / NULL), NOT the race field the stops layers use. Stored raw.
    "Black" in the arrest index means RACE = 'BLACK' regardless of the
    Hispanic flag; the definition is stated wherever the index is shown.
  * ARREST_DISTRICT: '1D'-'7D' plus NULL/'NA' — the same normalize_district
    used for stops handles all of it (trailing 'D' stripped, NA -> unmapped).
  * TYPE: 'Adult Arrest' vs 'Adult' (two vocabulary eras) — raw only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from database.stops import normalize_district, _raw_text

ARRESTS_LAYER_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/38"
)

ARREST_FIELDS = ",".join([
    "GLOBALID", "OBJECTID", "ARREST_NUMBER", "CCN",
    "TYPE", "YEAR", "DATE_", "HOUR",
    "AGE", "RACE", "ETHNICITY", "SEX",
    "CATEGORY", "DESCRIPTION",
    "ARREST_DISTRICT", "ARREST_PSA",
    "OFFENSE_DISTRICT", "OFFENSE_PSA",
    "DEFENDANT_DISTRICT", "DEFENDANT_PSA",
    "ARREST_LATITUDE", "ARREST_LONGITUDE",
])

ARREST_BATCHES_DDL = """
CREATE TABLE IF NOT EXISTS arrest_ingest_batches (
    batch_id            BIGSERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    fetched_rows        INTEGER,
    upserted_rows       INTEGER,
    distinct_globalids  INTEGER,
    note                TEXT
);
"""

ARRESTS_DDL = """
CREATE TABLE IF NOT EXISTS arrests (
    id                     SERIAL PRIMARY KEY,
    ingest_batch           BIGINT NOT NULL REFERENCES arrest_ingest_batches(batch_id),
    global_id              TEXT NOT NULL,
    arrest_number          TEXT,
    ccn                    TEXT,
    source_objectid        BIGINT,
    -- Person
    race                   TEXT,      -- RACE verbatim ('UNK'/'UNKNOWN' preserved)
    hispanic_raw           TEXT,      -- ETHNICITY verbatim (the Hispanic flag)
    sex                    TEXT,
    age                    TEXT,
    -- What
    arrest_type_raw        TEXT,      -- TYPE verbatim (two vocabulary eras)
    category               TEXT,
    description            TEXT,
    -- When
    arrest_datetime        TIMESTAMPTZ,
    year                   SMALLINT,  -- source YEAR field (authoritative)
    hour_raw               TEXT,
    -- Where (arrest location is the disparity geography; PSA is finer grain)
    arrest_district_raw    TEXT,
    district               SMALLINT,  -- normalized 1-7; NULL when unmappable
    district_unmapped      BOOLEAN NOT NULL DEFAULT FALSE,
    arrest_psa             TEXT,
    offense_district_raw   TEXT,
    offense_psa            TEXT,
    defendant_district_raw TEXT,
    defendant_psa          TEXT,
    arrest_latitude        DOUBLE PRECISION,
    arrest_longitude       DOUBLE PRECISION,
    ingested_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (global_id, ingest_batch)
);
CREATE INDEX IF NOT EXISTS idx_arrests_batch    ON arrests(ingest_batch);
CREATE INDEX IF NOT EXISTS idx_arrests_district ON arrests(district);
CREATE INDEX IF NOT EXISTS idx_arrests_year     ON arrests(year);
CREATE INDEX IF NOT EXISTS idx_arrests_race     ON arrests(race);
"""

ARREST_COLUMNS = (
    "ingest_batch",
    "global_id", "arrest_number", "ccn", "source_objectid",
    "race", "hispanic_raw", "sex", "age",
    "arrest_type_raw", "category", "description",
    "arrest_datetime", "year", "hour_raw",
    "arrest_district_raw", "district", "district_unmapped", "arrest_psa",
    "offense_district_raw", "offense_psa",
    "defendant_district_raw", "defendant_psa",
    "arrest_latitude", "arrest_longitude",
)


def _dt(epoch_ms: Any) -> Optional[datetime]:
    if epoch_ms in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(epoch_ms) / 1000, tz=timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _year(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v is not None else None
    except (ValueError, TypeError):
        return None


def transform_arrest(attrs: dict, ingest_batch: int) -> Optional[tuple]:
    """Map one ArcGIS arrest feature to an ``arrests`` row tuple."""
    gid = attrs.get("GLOBALID")
    if not gid:
        return None
    district, unmapped = normalize_district(attrs.get("ARREST_DISTRICT"))
    return (
        ingest_batch,
        str(gid),
        _raw_text(attrs.get("ARREST_NUMBER")),
        _raw_text(attrs.get("CCN")),
        attrs.get("OBJECTID"),
        attrs.get("RACE"),
        attrs.get("ETHNICITY"),
        attrs.get("SEX"),
        _raw_text(attrs.get("AGE")),
        attrs.get("TYPE"),
        attrs.get("CATEGORY"),
        attrs.get("DESCRIPTION"),
        _dt(attrs.get("DATE_")),
        _year(attrs.get("YEAR")),
        _raw_text(attrs.get("HOUR")),
        attrs.get("ARREST_DISTRICT"),
        district,
        unmapped,
        _raw_text(attrs.get("ARREST_PSA")),
        _raw_text(attrs.get("OFFENSE_DISTRICT")),
        _raw_text(attrs.get("OFFENSE_PSA")),
        _raw_text(attrs.get("DEFENDANT_DISTRICT")),
        _raw_text(attrs.get("DEFENDANT_PSA")),
        attrs.get("ARREST_LATITUDE"),
        attrs.get("ARREST_LONGITUDE"),
    )


_UPSERT_SQL = f"""
INSERT INTO arrests ({", ".join(ARREST_COLUMNS)})
VALUES ({", ".join(f"${i}" for i in range(1, len(ARREST_COLUMNS) + 1))})
ON CONFLICT (global_id, ingest_batch) DO UPDATE SET
    {", ".join(f"{c}=EXCLUDED.{c}" for c in ARREST_COLUMNS if c not in ("global_id", "ingest_batch"))},
    ingested_at=NOW()
"""


async def create_arrests_schema(conn) -> None:
    await conn.execute(ARREST_BATCHES_DDL)
    await conn.execute(ARRESTS_DDL)


async def start_batch(conn, note: Optional[str] = None) -> int:
    return await conn.fetchval(
        "INSERT INTO arrest_ingest_batches (note) VALUES ($1) RETURNING batch_id", note)


async def finish_batch(conn, batch_id: int, fetched_rows: int, upserted_rows: int,
                       distinct_globalids: int) -> None:
    await conn.execute(
        "UPDATE arrest_ingest_batches SET finished_at=NOW(), fetched_rows=$2, "
        "upserted_rows=$3, distinct_globalids=$4 WHERE batch_id=$1",
        batch_id, fetched_rows, upserted_rows, distinct_globalids)


async def upsert_arrests(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    await conn.executemany(_UPSERT_SQL, rows)
    return len(rows)


# ── Arrest representation (reuses the stops disparity pure functions) ─────────
ARREST_BLACK = "BLACK"                        # arrests RACE vocabulary
_ARREST_UNKNOWN = ("UNK", "UNKNOWN")          # BOTH unknown labels in the source


def arrest_cells_to_disparity_cells(rows: list[dict]) -> list[dict]:
    """Map arrest (district, race, n) cells onto the stops-engine cell contract:
    ethnicity 'Black' for RACE='BLACK', 'Unknown' for UNK/UNKNOWN/NULL, the raw
    label otherwise — so representation_by_district / citywide_representation /
    summarize_exclusions apply unchanged."""
    out = []
    for r in rows:
        race = r.get("race")
        if race is None or race in _ARREST_UNKNOWN:
            eth = "Unknown"
        elif race == ARREST_BLACK:
            eth = "Black"
        else:
            eth = race
        out.append({"district": r.get("district"), "ethnicity": eth, "n": r["n"]})
    return out


_ARREST_CELLS_SQL = """
SELECT district, race, COUNT(*) AS n FROM arrests
WHERE ingest_batch = $1 GROUP BY district, race
"""

_LATEST_ARREST_BATCH_SQL = """
SELECT MAX(batch_id) FROM arrest_ingest_batches WHERE finished_at IS NOT NULL
"""


async def arrest_representation_index(conn, *, batch_id: Optional[int] = None) -> dict:
    """Per-district arrest representation index + citywide, same definitions and
    baselines as the stop index (local + citywide population baselines, loud
    exclusions). 'Black' = RACE 'BLACK' regardless of the separate Hispanic flag."""
    from database.disparity import (
        _fetch_citywide_baseline, _fetch_district_meta,
        citywide_representation, representation_by_district, summarize_exclusions,
    )
    import asyncpg
    try:
        batch = batch_id if batch_id is not None else await conn.fetchval(_LATEST_ARREST_BATCH_SQL)
    except asyncpg.UndefinedTableError:
        raise RuntimeError(
            "arrest tables not created yet - run scripts/ingest_arrests.py first.")
    if batch is None:
        raise RuntimeError(
            "no finished arrest-ingest batch — run scripts/ingest_arrests.py first.")
    rows = await conn.fetch(_ARREST_CELLS_SQL, batch)
    cells = arrest_cells_to_disparity_cells([dict(r) for r in rows])
    baseline, baseline_info = await _fetch_citywide_baseline(conn)
    district_meta = await _fetch_district_meta(conn)
    return {
        "meta": {
            "grain": "arrests of persons",
            "ingest_batch": batch,
            "black_definition": "RACE = 'BLACK' (the separate Hispanic flag is not applied)",
            "citywide_black_pop_share": baseline,
            "baseline_source": baseline_info,
        },
        "districts": representation_by_district(cells, baseline, district_meta),
        "citywide": citywide_representation(cells, baseline),
        "exclusions": summarize_exclusions(cells),
    }
