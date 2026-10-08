"""Guarded upsert: MPD corrections to existing crime rows.

The ingest's ON CONFLICT (ccn, city) used to DO NOTHING, so MPD revisions of
already-stored records never landed. It is now a guarded DO UPDATE:

  (1) a single mutable-field correction applies (and logs old -> new);
  (2) an incoming NULL/empty value never overwrites a populated stored value;
  (3) a sync that would rewrite an anomalous share of matched rows trips the
      brake — a warning with a sample of the corrections is logged, corrections
      NOT written, insert-only fallback — and the explicit bulk bypass lets the
      same sync through;
  (4) identity (ccn, city) is never modified.

No live DB: the asyncpg pool is faked (repo pattern, tests/test_db_upsert_sql.py)
and the SQL is asserted structurally (repo pattern,
tests/test_pending_report_immutability.py). This public copy has no alert
email: where the app emailed an admin, it logs a warning.
"""

import logging
import re
from datetime import datetime, timezone

import pytest

from config import settings as _cfg
from database import db

_log = logging.getLogger(__name__)

REPORT_MS = 1_717_200_000_000  # 2024-06-01T00:00:00Z
REPORT_DT = datetime.fromtimestamp(REPORT_MS / 1000, tz=timezone.utc)


def arcgis_record(ccn="24001", **over):
    """An incoming ArcGIS attributes dict, consistent with stored_row()."""
    rec = {
        "CCN": ccn, "OFFENSE": "THEFT/OTHER", "METHOD": "OTHERS", "SHIFT": "DAY",
        "REPORT_DAT": REPORT_MS, "WARD": "5", "DISTRICT": "5D",
        "BLOCK": "100 - 199 BLOCK OF X ST", "NEIGHBORHOOD_CLUSTER": "cluster 21",
        "LATITUDE": 38.9, "LONGITUDE": -77.0,
    }
    rec.update(over)
    return rec


def stored_row(ccn="24001", **over):
    """The already-stored DB row for the same record (no effective diff)."""
    row = {
        "ccn": ccn, "offense": "THEFT/OTHER", "method": "OTHERS", "shift": "DAY",
        "report_dat": REPORT_DT, "year": 2024, "ward": "5", "district": "5D",
        "block": "100 - 199 BLOCK OF X ST", "neighborhood_cluster": "cluster 21",
        "latitude": 38.9, "longitude": -77.0,
    }
    row.update(over)
    return row


class FakeConn:
    """Captures the pre-read fetch and the executemany write. SKIPPED live DB —
    guard semantics are asserted on the SQL text + the Python mirror, exactly
    like the repo's other upsert tests."""

    def __init__(self, existing_rows):
        self.existing_rows = existing_rows
        self.fetches = []
        self.writes = []

    async def fetch(self, sql, *params):
        self.fetches.append((sql, params))
        return self.existing_rows

    async def executemany(self, sql, rows):
        self.writes.append((sql, list(rows)))


def _install_fake_pool(monkeypatch, existing_rows):
    conn = FakeConn(existing_rows)

    class _Acquire:
        async def __aenter__(self):
            return conn

        async def __aexit__(self, *exc):
            return None

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def fake_get_pool():
        return _Pool()

    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    monkeypatch.setattr(db, "_has_postgis", False)  # plain variant; same guard clause
    _log.info("SKIPPED live DB — asyncpg pool faked (no DATABASE_URL in unit tests)")
    return conn


@pytest.fixture
def anomaly_email():
    """The public copy has no alert email; the brake logs a warning instead."""
    return None


# ── (1) a single mutable-field correction applies ─────────────────────────────

async def test_single_field_correction_applies_and_logs(monkeypatch, anomaly_email, caplog):
    conn = _install_fake_pool(monkeypatch, [stored_row(offense="THEFT/OTHER")])

    with caplog.at_level(logging.INFO, logger="database.db"):
        n = await db.upsert_records([arcgis_record(OFFENSE="ROBBERY")], city="dc")

    assert n == 1
    sql, rows = conn.writes[0]
    assert "ON CONFLICT (ccn, city) DO UPDATE SET" in sql, (
        "a routine correction must go through the guarded DO UPDATE, not DO NOTHING"
    )
    assert rows[0][2] == "ROBBERY"  # corrected offense reaches the write
    # Routine single-record corrections log old -> new but never page the admin.
    assert any(
        "correction" in r.message and "offense" in r.message
        and "THEFT/OTHER" in r.message and "ROBBERY" in r.message
        for r in caplog.records
    ), "the applied correction must be change-logged old -> new"


async def test_corrections_below_floor_never_trip_brake(monkeypatch, anomaly_email):
    # 3 of 4 matched rows change: 75% is far past the 10% gate, but 3 < the
    # 50-row floor — high-percentage tiny syncs are routine corrections.
    existing = [stored_row(ccn=f"24{i:03d}") for i in range(4)]
    conn = _install_fake_pool(monkeypatch, existing)
    incoming = [arcgis_record(ccn=f"24{i:03d}", OFFENSE="ROBBERY") for i in range(3)]
    incoming.append(arcgis_record(ccn="24003"))

    await db.upsert_records(incoming, city="dc")

    assert "DO UPDATE SET" in conn.writes[0][0]


# ── (2) incoming NULL/empty never overwrites populated data ───────────────────

def test_effective_update_keeps_populated_values_on_null_or_empty():
    stored = stored_row()
    incoming = dict(
        zip(("ccn", "city") + db._MUTABLE_FIELDS,
            db._arcgis_record_to_row(
                arcgis_record(OFFENSE=None, BLOCK="", LATITUDE=None), city="dc"
            ))
    )

    eff = db._effective_update(stored, incoming)

    assert eff["offense"] == "THEFT/OTHER"          # NULL kept stored value
    assert eff["block"] == "100 - 199 BLOCK OF X ST"  # empty string kept stored value
    # Half-missing coordinate pair keeps BOTH stored coordinates.
    assert (eff["latitude"], eff["longitude"]) == (38.9, -77.0)


def test_year_is_derived_only_from_report_dat():
    fields = ("ccn", "city") + db._MUTABLE_FIELDS

    # Valid epoch-ms timestamp → year matches the timestamp's year.
    ok = dict(zip(fields, db._arcgis_record_to_row(arcgis_record(), city="dc")))
    assert ok["report_dat"] == REPORT_DT
    assert ok["year"] == REPORT_DT.year == 2024

    # Missing/unparseable REPORT_DAT → BOTH report_dat and year are NULL, so year
    # can never diverge from report_dat (e.g. a correction moving year across a
    # boundary while the stored timestamp stays put). The "2023-01-01" case is the
    # year-prefixed string the old fallback would have parsed into year=2023.
    for bad in (None, "", "not-a-date", "2023-01-01"):
        row = dict(zip(fields, db._arcgis_record_to_row(arcgis_record(REPORT_DAT=bad), city="dc")))
        assert row["report_dat"] is None, bad
        assert row["year"] is None, bad


async def test_degraded_record_produces_no_corrections(monkeypatch, anomaly_email):
    conn = _install_fake_pool(monkeypatch, [stored_row()])
    degraded = arcgis_record(
        OFFENSE=None, METHOD="", SHIFT=None, WARD="", DISTRICT=None,
        BLOCK="", NEIGHBORHOOD_CLUSTER=None, LATITUDE=None, LONGITUDE=None,
    )

    _, corrections = await db._compute_corrections(
        conn, [db._arcgis_record_to_row(degraded, "dc")], "dc"
    )

    assert corrections == {}, (
        "a fully degraded ArcGIS record must propose zero changes to a populated row"
    )


def _set_clause(sql: str) -> str:
    m = re.search(r"DO UPDATE SET(.*?)\nWHERE", sql, re.DOTALL)
    assert m, "guarded upsert must have a DO UPDATE SET … WHERE shape"
    return m.group(1)


@pytest.mark.parametrize("sql_name", ["_UPSERT_SQL_POSTGIS", "_UPSERT_SQL_PLAIN"])
def test_sql_null_and_empty_guards_on_every_field(sql_name):
    sql = getattr(db, sql_name)
    set_clause = _set_clause(sql)
    for f in db._TEXT_FIELDS:
        assert f"{f} = COALESCE(NULLIF(EXCLUDED.{f}, ''), crimes.{f})" in set_clause, (
            f"text field {f} must NULL/empty-coalesce against the stored value"
        )
    for f in ("report_dat", "year"):
        assert f"{f} = COALESCE(EXCLUDED.{f}, crimes.{f})" in set_clause
    for f in ("latitude", "longitude"):
        assert (
            f"{f} = CASE WHEN EXCLUDED.latitude IS NOT NULL AND "
            f"EXCLUDED.longitude IS NOT NULL THEN EXCLUDED.{f} ELSE crimes.{f} END"
        ) in set_clause, "coordinates must only move as a complete pair"


# ── (3) anomaly brake + bulk bypass ───────────────────────────────────────────

def _anomalous_sync():
    """10 matched rows, 6 of which the sync would rewrite."""
    existing = [stored_row(ccn=f"24{i:03d}") for i in range(10)]
    incoming = [
        arcgis_record(ccn=f"24{i:03d}", OFFENSE="ARSON" if i < 6 else "THEFT/OTHER")
        for i in range(10)
    ]
    return existing, incoming


async def test_brake_trips_alert_fires_and_corrections_not_written(
    monkeypatch, anomaly_email, caplog
):
    existing, incoming = _anomalous_sync()
    conn = _install_fake_pool(monkeypatch, existing)
    # Scale the floor to the fixture (prod default 50 is pinned separately below).
    monkeypatch.setattr(_cfg, "UPSERT_BRAKE_MIN_CHANGED", 5)

    with caplog.at_level(logging.WARNING, logger="database.db"):
        await db.upsert_records(incoming, city="dc")

    assert any("sample of the withheld corrections" in r.message for r in caplog.records)
    sql, rows = conn.writes[0]
    assert "DO NOTHING" in sql and "DO UPDATE" not in sql, (
        "brake must fall back to legacy insert-only — corrections NOT written"
    )
    assert len(rows) == 10  # new-row inserts still happen, as before the change
    assert any("anomaly brake tripped" in r.message for r in caplog.records)


async def test_bulk_bypass_lets_anomalous_sync_through(monkeypatch, anomaly_email):
    existing, incoming = _anomalous_sync()
    conn = _install_fake_pool(monkeypatch, existing)
    monkeypatch.setattr(_cfg, "UPSERT_BRAKE_MIN_CHANGED", 5)

    await db.upsert_records(incoming, city="dc", allow_bulk_rewrite=True)

    assert "DO UPDATE SET" in conn.writes[0][0], (
        "explicit --allow-bulk-rewrite must apply the corrections"
    )


def test_brake_defaults_are_conservative():
    # Documented defaults: >10% of matched rows AND >=50 rows.
    # Pin them so a quiet default change can't disarm the brake.
    assert _cfg.UPSERT_BRAKE_MAX_CHANGED_PCT == 10.0
    assert _cfg.UPSERT_BRAKE_MIN_CHANGED == 50


# ── (4) identity (ccn, city) is never modified ────────────────────────────────

@pytest.mark.parametrize("sql_name", ["_UPSERT_SQL_POSTGIS", "_UPSERT_SQL_PLAIN"])
def test_set_clause_never_assigns_ccn_or_city(sql_name):
    set_clause = _set_clause(getattr(db, sql_name))
    assert not re.search(r"\bccn\s*=", set_clause), "DO UPDATE must never assign ccn"
    assert not re.search(r"\bcity\s*=", set_clause), "DO UPDATE must never assign city"


def test_mutable_fields_exclude_identity():
    assert "ccn" not in db._MUTABLE_FIELDS
    assert "city" not in db._MUTABLE_FIELDS
    assert "id" not in db._MUTABLE_FIELDS
    # The Python mirror can never propose an identity change either.
    eff = db._effective_update(stored_row(), dict.fromkeys(("ccn", "city") + db._MUTABLE_FIELDS))
    assert set(eff) == set(db._MUTABLE_FIELDS)
