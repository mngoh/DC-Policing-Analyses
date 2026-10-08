"""MPD Stop Data — table DDL, district normalization, batch versioning, and upsert.

Additive numerator source for the accountability tool. This module is NOT wired
into app startup (ensure_schema): the DDL is created on demand by the manual
ingest script (scripts/ingest_stops.py) so ingestion can be run and inspected
before it is ever scheduled. Nothing here touches crime_index.py, the report
modules, the frontend, or the existing crimes table.

Source: MPD "Stop Data", DC ArcGIS Public_Safety_WebMercator — TWO layers:
  * layer 41: 2023-present (~229k rows)
  * layer 45: 2019-07-22 through 2022-12-31 (~280k rows). Nothing exists before
    2019-07-22: MPD only began NEAR Act stop-data collection then, so the stops
    record can never reach the crime table's 2008 depth (collection gap, not a
    fetch gap).
Layer 45 schema drift (verified against the live service 2026-07-12) is handled
here: STOP_LOCATION (45) maps to the same stop_location_block column as
STOP_LOCATION_BLOCK (41); ARREST_CHARGES_CATEGORY does not exist on 45 (lands as
NULL in arrest_charges_cat_raw, a raw-only column anyway). STOP_TYPE has the same
four live values on both layers; districts are '1'-'7' plus ''/'NA' (both
normalize to unmapped). One 45-only wrinkle: ETHNICITY contains literal SQL NULLs
(~1.8k rows) and both 'Hispanic' and 'Hispanic/Latino' labels — stored verbatim,
like everything else.

Batch versioning (folds in the resolution of open question #1)
--------------------------------------------------------------
Each ingest run is recorded as a BATCH (a row in ``stop_ingest_batches``) and
every stop row is tagged with its ``ingest_batch``. The conflict key is
``(global_id, ingest_batch)``, so a batch is a self-consistent *snapshot* of one
MPD release:
  * Re-fetching within the SAME batch is idempotent (upsert on the composite key).
  * A NEW run = a NEW batch = a fresh snapshot. This is deliberate: MPD refreshes
    Stop Data only ~SEMI-ANNUALLY and may regenerate GLOBALIDs on republish, so we
    do NOT bet on cross-release GLOBALID stability. Instead each release lands as
    its own snapshot; "current" stops are ``ingest_batch = latest batch for the
    layer``, and drift between releases is measured by GLOBALID overlap across
    consecutive batches (recorded in ``stop_ingest_batches``).
GLOBALID (not CCN_ANONYMIZED) is the per-row key inside a batch: it is verified
row-unique, whereas CCN is one-per-stop-event and shared by multi-person stops.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# Layer 41 = current Stop Data (2023-present). Layer 45 = 2019-07-22 to 2022.
_LAYER_URL_TMPL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/{layer}"
)
STOPS_LAYER_URL = _LAYER_URL_TMPL.format(layer=41)   # kept name: layer-41 URL
CURRENT_LAYER = 41
LAYER_URLS = {41: STOPS_LAYER_URL, 45: _LAYER_URL_TMPL.format(layer=45)}

# Source fields pulled (a curated numerator-focused subset, NOT all ~70 columns —
# the search / protective-pat-down boolean matrix is left for a later analytical
# pass). ARREST_*/TICKET_*/WARNING_* are ingested RAW but are known garbage-encoded
# (literal "NULL" strings, placeholder 1/1/2023 dates); do not build logic on them.
_SHARED_FIELDS = [
    "GLOBALID", "OBJECTID", "CCN_ANONYMIZED",
    "ETHNICITY", "GENDER", "AGE",
    "STOP_DISTRICT",
    "STOP_TYPE", "DATETIME", "STOP_DURATION_MINS",
    "ARREST_DATE",
    "TICKETS_ISSUED", "WARNINGS_ISSUED", "TICKET_COUNT", "WARNING_COUNT",
    # Person-search matrix (identical names on 41 and 45; verified live
    # 2026-07-12). The four flags are clean numeric booleans (1.0/0.0, no
    # NULLs). The three *_SEIZED fields are messy multi-value free text with
    # era-varying vocabulary — stored RAW; the outcome-test engine parses them
    # with a unit-tested tokenizer. There is NO seizure-outcome field for
    # probable-cause person searches in the source (property has one; person
    # does not), so that search type gets search rates only.
    "PERSON_SEARCH_CONSENT", "PERSON_SEARCH_PROB_CAUSE",
    "PERSON_SEARCH_WARRANT", "PERSON_PROT_PAT_DOWN",
    "PERSON_SRCH_OBJ_SEIZED_CON", "PERSON_SRCH_OBJ_SEIZED_WAR",
    "PER_PRO_PAT_DOWN_OBJ_SEIZE",
]
# Per-layer drift: 45 names the block-level location STOP_LOCATION and has no
# ARREST_CHARGES_CATEGORY (see module docstring).
STOP_FIELDS = ",".join(_SHARED_FIELDS + ["STOP_LOCATION_BLOCK", "ARREST_CHARGES_CATEGORY"])
STOP_FIELDS_45 = ",".join(_SHARED_FIELDS + ["STOP_LOCATION"])
LAYER_FIELDS = {41: STOP_FIELDS, 45: STOP_FIELDS_45}

# ── Schema (mirrors the style of the crimes table in database/db.py) ──────────
# One row per ingest run. A batch is a point-in-time snapshot of an MPD release.
STOP_BATCHES_DDL = """
CREATE TABLE IF NOT EXISTS stop_ingest_batches (
    batch_id            BIGSERIAL PRIMARY KEY,
    source_layer        SMALLINT NOT NULL,        -- 41=2023-present, 45=2019-2022
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    fetched_rows        INTEGER,                  -- rows returned by ArcGIS
    upserted_rows       INTEGER,                  -- rows written for this batch
    distinct_globalids  INTEGER,                  -- for cross-batch drift comparison
    note                TEXT
);
"""

STOPS_DDL = """
CREATE TABLE IF NOT EXISTS stops (
    id                     SERIAL PRIMARY KEY,
    ingest_batch           BIGINT NOT NULL REFERENCES stop_ingest_batches(batch_id),  -- snapshot this row belongs to
    -- Source identity
    global_id              TEXT NOT NULL,        -- ArcGIS GLOBALID (row-unique GUID within a batch)
    ccn_anonymized         TEXT,                 -- anonymized complaint number; ONE per stop, NOT row-unique
    source_objectid        BIGINT,               -- ArcGIS OBJECTID; transient, not stable across refreshes
    source_layer           SMALLINT NOT NULL DEFAULT 41,  -- 41=2023-present, 45=2019-2022 (later)
    -- Person demographics
    ethnicity              TEXT,                 -- ETHNICITY; "Unknown" kept as a real value, never NULL-collapsed
    gender                 TEXT,                 -- GENDER
    age                    TEXT,                 -- AGE (source is text)
    -- Geography (district is the ONLY geography — no coordinates, no ward)
    stop_district_raw      TEXT,                 -- STOP_DISTRICT verbatim ('7','7D','NA', or null)
    district               SMALLINT,             -- normalized 1-7 (trailing 'D' stripped); NULL when unmappable
    district_unmapped      BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE when raw was 'NA'/null/unparseable
    stop_location_block    TEXT,                 -- STOP_LOCATION_BLOCK (free text; only sub-district geo)
    -- Stop attributes
    stop_type              TEXT,                 -- STOP_TYPE (clean: ticket / non-ticket)
    stop_datetime          TIMESTAMPTZ,          -- DATETIME
    year                   SMALLINT,             -- derived from DATETIME (mirrors crimes.year)
    stop_duration_mins     DOUBLE PRECISION,     -- STOP_DURATION_MINS
    -- Outcome columns: GARBAGE-ENCODED in source (literal "NULL" strings,
    -- placeholder 1/1/2023 dates). Stored raw for fidelity only — DO NOT build
    -- logic on them (see findings doc).
    arrest_date_raw        TEXT,
    arrest_charges_cat_raw TEXT,
    tickets_issued_raw     TEXT,
    warnings_issued_raw    TEXT,
    ticket_count_raw       TEXT,
    warning_count_raw      TEXT,
    -- Person-search matrix (added for the outcome test; NULL on batches
    -- ingested before these fields were pulled). Flags are normalized booleans;
    -- seizure outcomes are raw text parsed by the engine.
    search_consent         BOOLEAN,
    search_prob_cause      BOOLEAN,
    search_warrant         BOOLEAN,
    pat_down               BOOLEAN,
    seized_consent_raw     TEXT,
    seized_warrant_raw     TEXT,
    seized_pat_down_raw    TEXT,
    -- Meta
    ingested_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Row identity is (global_id, ingest_batch): unique within a snapshot, and a
    -- new snapshot re-lands the same source row rather than silently colliding.
    UNIQUE (global_id, ingest_batch)
);
CREATE INDEX IF NOT EXISTS idx_stops_batch     ON stops(ingest_batch);
CREATE INDEX IF NOT EXISTS idx_stops_district  ON stops(district);
CREATE INDEX IF NOT EXISTS idx_stops_year      ON stops(year);
CREATE INDEX IF NOT EXISTS idx_stops_ethnicity ON stops(ethnicity);
CREATE INDEX IF NOT EXISTS idx_stops_ccn       ON stops(ccn_anonymized);
"""


def normalize_district(raw: Any) -> tuple[Optional[int], bool]:
    """Normalize a STOP_DISTRICT code to ``(district, unmapped)``.

    The source double-encodes districts as both ``'7'`` and ``'7D'``; both mean
    the Seventh District. Strip an optional trailing ``D`` and map to the integer
    1-7. Anything else — ``'NA'``, null, blank, out-of-range, or unparseable —
    returns ``(None, True)``: we flag it rather than guess a district.

    This is the one piece of the ingest with real branching; it is unit-tested.
    """
    if raw is None:
        return None, True
    s = str(raw).strip().upper()
    if s in ("", "NA"):
        return None, True
    if s.endswith("D"):
        s = s[:-1]
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 7:
            return n, False
    return None, True


def _raw_text(v: Any) -> Optional[str]:
    """Store a garbage-encoded raw outcome value as text, faithfully.

    Layer 41 serves these as strings; layer 45 serves TICKET_COUNT/WARNING_COUNT
    (and friends) as numbers, which asyncpg refuses to bind to TEXT. str() the
    value — the columns are raw-only fidelity storage either way.
    """
    return None if v is None else str(v)


def _dt_and_year(epoch_ms: Any) -> tuple[Optional[datetime], Optional[int]]:
    """ArcGIS DATETIME is epoch milliseconds (UTC). Return (datetime, year)."""
    if epoch_ms in (None, ""):
        return None, None
    try:
        dt = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=timezone.utc)
        return dt, dt.year
    except (ValueError, TypeError, OverflowError, OSError):
        return None, None


# Column order for both the INSERT and the row tuple. ``ingest_batch`` leads
# because it (with global_id) is the row identity.
STOP_COLUMNS = (
    "ingest_batch",
    "global_id", "ccn_anonymized", "source_objectid", "source_layer",
    "ethnicity", "gender", "age",
    "stop_district_raw", "district", "district_unmapped", "stop_location_block",
    "stop_type", "stop_datetime", "year", "stop_duration_mins",
    "arrest_date_raw", "arrest_charges_cat_raw",
    "tickets_issued_raw", "warnings_issued_raw", "ticket_count_raw", "warning_count_raw",
    "search_consent", "search_prob_cause", "search_warrant", "pat_down",
    "seized_consent_raw", "seized_warrant_raw", "seized_pat_down_raw",
)


def _flag(v: Any) -> Optional[bool]:
    """Normalize a source search flag (numeric 1.0/0.0 on both layers) to bool.
    None or unparseable stays None — never guessed to False."""
    if v is None:
        return None
    try:
        return bool(int(float(v)))
    except (ValueError, TypeError):
        return None


def transform_stop(attrs: dict, source_layer: int, ingest_batch: int) -> Optional[tuple]:
    """Map one ArcGIS Stop-Data feature to a ``stops`` row tuple for ``ingest_batch``.

    Returns ``None`` (skipped) only when GLOBALID — half the row identity — is
    missing, which should never happen for this layer. Race is passed through
    verbatim so the literal ``"Unknown"`` category is preserved and never dropped.
    """
    gid = attrs.get("GLOBALID")
    if not gid:
        return None
    district, unmapped = normalize_district(attrs.get("STOP_DISTRICT"))
    stop_dt, year = _dt_and_year(attrs.get("DATETIME"))
    # Same column, per-layer source field: 45 calls the block-level location
    # STOP_LOCATION; 41 calls it STOP_LOCATION_BLOCK.
    location = (attrs.get("STOP_LOCATION") if source_layer == 45
                else attrs.get("STOP_LOCATION_BLOCK"))
    return (
        ingest_batch,
        str(gid),
        attrs.get("CCN_ANONYMIZED"),
        attrs.get("OBJECTID"),
        source_layer,
        attrs.get("ETHNICITY"),           # "Unknown" preserved as-is
        attrs.get("GENDER"),
        attrs.get("AGE"),
        attrs.get("STOP_DISTRICT"),        # raw
        district,
        unmapped,
        location,
        attrs.get("STOP_TYPE"),
        stop_dt,
        year,
        attrs.get("STOP_DURATION_MINS"),
        _raw_text(attrs.get("ARREST_DATE")),          # raw / unreliable
        _raw_text(attrs.get("ARREST_CHARGES_CATEGORY")),
        _raw_text(attrs.get("TICKETS_ISSUED")),
        _raw_text(attrs.get("WARNINGS_ISSUED")),
        _raw_text(attrs.get("TICKET_COUNT")),
        _raw_text(attrs.get("WARNING_COUNT")),
        _flag(attrs.get("PERSON_SEARCH_CONSENT")),
        _flag(attrs.get("PERSON_SEARCH_PROB_CAUSE")),
        _flag(attrs.get("PERSON_SEARCH_WARRANT")),
        _flag(attrs.get("PERSON_PROT_PAT_DOWN")),
        _raw_text(attrs.get("PERSON_SRCH_OBJ_SEIZED_CON")),
        _raw_text(attrs.get("PERSON_SRCH_OBJ_SEIZED_WAR")),
        _raw_text(attrs.get("PER_PRO_PAT_DOWN_OBJ_SEIZE")),
    )


_UPSERT_SQL = f"""
INSERT INTO stops ({", ".join(STOP_COLUMNS)})
VALUES ({", ".join(f"${i}" for i in range(1, len(STOP_COLUMNS) + 1))})
ON CONFLICT (global_id, ingest_batch) DO UPDATE SET
    {", ".join(f"{c}=EXCLUDED.{c}" for c in STOP_COLUMNS if c not in ("global_id", "ingest_batch"))},
    ingested_at=NOW()
"""


# Columns added after the first production ingest; CREATE TABLE IF NOT EXISTS
# skips an existing table, so they are also applied as idempotent ALTERs.
_STOPS_MIGRATIONS = """
ALTER TABLE stops
    ADD COLUMN IF NOT EXISTS search_consent      BOOLEAN,
    ADD COLUMN IF NOT EXISTS search_prob_cause   BOOLEAN,
    ADD COLUMN IF NOT EXISTS search_warrant      BOOLEAN,
    ADD COLUMN IF NOT EXISTS pat_down            BOOLEAN,
    ADD COLUMN IF NOT EXISTS seized_consent_raw  TEXT,
    ADD COLUMN IF NOT EXISTS seized_warrant_raw  TEXT,
    ADD COLUMN IF NOT EXISTS seized_pat_down_raw TEXT
"""


async def create_stops_schema(conn) -> None:
    """Create the batch + stops tables and indexes (idempotent)."""
    await conn.execute(STOP_BATCHES_DDL)
    await conn.execute(STOPS_DDL)
    await conn.execute(_STOPS_MIGRATIONS)


async def start_batch(conn, source_layer: int = CURRENT_LAYER, note: Optional[str] = None) -> int:
    """Open a new ingest batch (snapshot) and return its ``batch_id``."""
    return await conn.fetchval(
        "INSERT INTO stop_ingest_batches (source_layer, note) VALUES ($1, $2) RETURNING batch_id",
        source_layer, note,
    )


async def finish_batch(conn, batch_id: int, fetched_rows: int, upserted_rows: int,
                       distinct_globalids: int) -> None:
    """Record completion stats for a batch (for drift comparison across releases)."""
    await conn.execute(
        "UPDATE stop_ingest_batches SET finished_at=NOW(), fetched_rows=$2, "
        "upserted_rows=$3, distinct_globalids=$4 WHERE batch_id=$1",
        batch_id, fetched_rows, upserted_rows, distinct_globalids,
    )


async def upsert_stops(conn, rows: list[tuple]) -> int:
    """Upsert stop rows for one batch on (global_id, ingest_batch). Returns count.

    Idempotent WITHIN a batch (re-fetching the same snapshot updates in place). A
    new batch is a separate snapshot — see the module docstring / findings doc.
    """
    if not rows:
        return 0
    await conn.executemany(_UPSERT_SQL, rows)
    return len(rows)
