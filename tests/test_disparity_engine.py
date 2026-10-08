"""Tests for the disparity engine — the one module where a wrong number is THE failure.

Exercises the pure aggregation over synthetic person-rows (no DB/network; the
suite runs with DATABASE_URL disabled). Covers: the index math, that exclusions
are actually applied to the denominator, that the baseline is citywide and not
self-referential, and that the outcome split uses STOP_TYPE only.
"""
import pytest

from database.disparity import (
    BLACK,
    NIGHT_HOURS,
    NO_TICKET_STOP_TYPE,
    UNKNOWN_RACE,
    _cells_sql,
    _time_filter_sql,
    citywide_representation,
    group_stop_rows,
    outcome_split_by_district,
    outcome_split_exclusions,
    representation_by_district,
    summarize_exclusions,
)


def _rows(district, ethnicity, stop_type, n):
    """n identical person-rows (one row = one person, the grain)."""
    return [{"district": district, "ethnicity": ethnicity, "stop_type": stop_type}
            for _ in range(n)]


# ── 1. Index math on a small fixture ──────────────────────────────────────────
def test_index_math_basic():
    cells = group_stop_rows(
        _rows(7, BLACK, "Non-ticket Stop", 90) + _rows(7, "White", "Ticket Only", 10)
    )
    meta = {7: {"police_district": "7D", "black_pop_share": 0.865}}
    rows = representation_by_district(cells, 0.45, meta)

    assert len(rows) == 1
    r = rows[0]
    assert r["black_stops"] == 90
    assert r["known_race_stops"] == 100
    assert r["black_stop_share"] == pytest.approx(0.90)
    assert r["citywide_black_pop_share"] == 0.45
    assert r["representation_index"] == pytest.approx(2.0)   # 0.90 / 0.45
    assert r["district_black_pop_share"] == 0.865            # context, not the baseline
    assert r["police_district"] == "7D"


# ── 2. Exclusions are APPLIED to the denominator (and reported) ────────────────
def test_unknown_race_and_unmapped_excluded_from_denominator():
    raw = (
        _rows(1, BLACK, "Non-ticket Stop", 90)     # counts
        + _rows(1, "White", "Ticket Only", 10)     # counts (known, non-Black)
        + _rows(1, UNKNOWN_RACE, "Non-ticket Stop", 50)   # excluded: unknown race
        + _rows(None, BLACK, "Non-ticket Stop", 30)       # excluded: unmapped district
    )
    cells = group_stop_rows(raw)
    rows = representation_by_district(cells, 0.45, {})

    # Only district 1 produces a row; the unmapped rows never form a district.
    assert [r["district"] for r in rows] == [1]
    r = rows[0]
    assert r["known_race_stops"] == 100     # 90 + 10, NOT 150 — Unknown excluded
    assert r["black_stops"] == 90
    assert r["unknown_race_stops"] == 50    # excluded but reported per-district

    excl = summarize_exclusions(cells)
    assert excl["total_stops"] == 180
    assert excl["unknown_race_stops"] == 50
    assert excl["unmapped_district_stops"] == 30
    # Reported figures rest on the mapped known-race rows: 100 of 180.
    assert excl["reported_coverage_pct"] == pytest.approx(100 / 180 * 100, rel=1e-3)


def test_null_ethnicity_treated_as_unknown():
    cells = group_stop_rows(
        _rows(3, BLACK, "Non-ticket Stop", 40) + _rows(3, None, "Non-ticket Stop", 60)
    )
    r = representation_by_district(cells, 0.45, {})[0]
    assert r["known_race_stops"] == 40      # SQL NULL race is not a known race
    assert r["unknown_race_stops"] == 60


# ── 3. Baseline is CITYWIDE, not self-referential ─────────────────────────────
def test_baseline_is_citywide_not_self_referential():
    # Near-single-race district: 95% Black stops. Its OWN pop share is ~0.865.
    cells = group_stop_rows(
        _rows(7, BLACK, "Non-ticket Stop", 950) + _rows(7, "White", "Ticket Only", 50)
    )
    citywide_baseline = 0.4326
    own_pop_share = 0.865
    r = representation_by_district(
        cells, citywide_baseline, {7: {"police_district": "7D", "black_pop_share": own_pop_share}}
    )[0]

    # Against the citywide baseline the signal is loud (~2.2x), NOT the ~1.0 you
    # get from a self-referential comparison.
    assert r["representation_index"] == pytest.approx(0.95 / citywide_baseline, rel=1e-6)
    assert r["representation_index"] > 2.0
    self_ref = 0.95 / own_pop_share          # ~1.098 — the meaningless comparison
    assert abs(r["representation_index"] - self_ref) > 0.5   # provably NOT self-referential


def test_citywide_summary_index():
    # Mirrors the real citywide figures (Black 167,647 of 213,336 known-race).
    cells = group_stop_rows(
        _rows(1, BLACK, "Non-ticket Stop", 167647)
        + _rows(1, "White", "Ticket Only", 45689)   # 213,336 - 167,647 known non-Black
        + _rows(1, UNKNOWN_RACE, "Non-ticket Stop", 15728)
    )
    summary = citywide_representation(cells, 0.432639)
    assert summary["black_stops"] == 167647
    assert summary["known_race_stops"] == 213336
    assert summary["black_stop_share"] == pytest.approx(167647 / 213336)
    assert summary["representation_index"] == pytest.approx(1.816, rel=1e-3)


# ── 3b. LOCAL baseline index (primary per-district metric) ────────────────────
def test_local_index_math():
    # 90% Black stops in a district that is 30% Black locally -> 3.0x local.
    cells = group_stop_rows(
        _rows(2, BLACK, "Non-ticket Stop", 90) + _rows(2, "White", "Ticket Only", 10)
    )
    r = representation_by_district(
        cells, 0.4326, {2: {"police_district": "2D", "black_pop_share": 0.30}}
    )[0]
    assert r["black_stop_share"] == pytest.approx(0.90)
    assert r["local_black_pop_share"] == 0.30
    assert r["local_representation_index"] == pytest.approx(3.0)   # 0.90 / 0.30
    assert r["black_stops"] == 90                                  # N the local ratio rests on
    assert r["known_race_stops"] == 100


def test_local_index_is_local_not_citywide():
    # A high-Black district (~86% pop, ~95% stops) -> ~1.1x LOCAL, but ~2.2x citywide.
    # A low-Black district (~8% pop, ~53% stops) -> high LOCAL, but only ~1.2x citywide.
    citywide = 0.4326
    cells = group_stop_rows(
        _rows(7, BLACK, "Non-ticket Stop", 955) + _rows(7, "White", "Ticket Only", 45)
        + _rows(2, BLACK, "Non-ticket Stop", 532) + _rows(2, "White", "Ticket Only", 468)
    )
    meta = {7: {"police_district": "7D", "black_pop_share": 0.865},
            2: {"police_district": "2D", "black_pop_share": 0.08}}
    rows = {r["district"]: r for r in representation_by_district(cells, citywide, meta)}

    high = rows[7]
    # Local: near ~1.0 in the majority-Black district (proves it's self-referential-local).
    assert high["local_representation_index"] == pytest.approx(0.955 / 0.865, rel=1e-6)
    assert high["local_representation_index"] < 1.2
    # Citywide (secondary) is loud here and clearly different from local.
    assert high["representation_index"] == pytest.approx(0.955 / citywide, rel=1e-6)
    assert high["representation_index"] > 2.0
    assert high["local_representation_index"] != pytest.approx(high["representation_index"])

    low = rows[2]
    # Local is loud in the low-Black district; citywide is muted -> opposite shape.
    assert low["local_representation_index"] == pytest.approx(0.532 / 0.08, rel=1e-6)
    assert low["local_representation_index"] > 6.0
    assert low["representation_index"] == pytest.approx(0.532 / citywide, rel=1e-6)
    assert low["representation_index"] < 1.3
    # The whole point: local flags the low-Black district hardest, citywide the high one.
    assert low["local_representation_index"] > high["local_representation_index"]
    assert high["representation_index"] > low["representation_index"]


def test_citywide_index_unchanged_alongside_local():
    # Adding the local metric must not disturb the citywide per-district index.
    cells = group_stop_rows(
        _rows(4, BLACK, "Non-ticket Stop", 70) + _rows(4, "White", "Ticket Only", 30)
    )
    r = representation_by_district(
        cells, 0.45, {4: {"police_district": "4D", "black_pop_share": 0.486}}
    )[0]
    assert r["representation_index"] == pytest.approx(0.70 / 0.45)   # citywide, unchanged
    assert r["citywide_black_pop_share"] == 0.45
    assert r["local_representation_index"] == pytest.approx(0.70 / 0.486)


# ── 4. Outcome split: STOP_TYPE only, Harbor excluded, tickets = enforcement ──
def test_outcome_split_excludes_harbor_and_counts_tickets_as_enforcement():
    raw = (
        # District 5, Black street stops: 30 no-enforcement of 45 street total.
        _rows(5, BLACK, "Non-ticket Stop", 30)
        + _rows(5, BLACK, "Ticket Only", 10)               # enforcement
        + _rows(5, BLACK, "Ticket and Non-ticket Stop", 5) # enforcement (a ticket issued)
        + _rows(5, BLACK, "Harbor", 8)                     # EXCLUDED from num + denom
        # District 5, non-Black known: 10 no-enforcement of 20 street.
        + _rows(5, "White", "Non-ticket Stop", 10)
        + _rows(5, "Hispanic/Latino", "Ticket Only", 10)
        + _rows(5, "White", "Harbor", 3)                   # EXCLUDED
        # Unknown race must not enter either group.
        + _rows(5, UNKNOWN_RACE, "Non-ticket Stop", 100)
    )
    cells = group_stop_rows(raw)
    r = outcome_split_by_district(cells, {5: {"police_district": "5D"}})[0]

    assert r["is_preliminary"] is True
    b = r["black"]
    # Harbor (8) is in NEITHER numerator nor denominator.
    assert b["street_total"] == 45                   # 30 + 10 + 5, Harbor excluded
    assert b["no_enforcement_stops"] == 30           # only 'Non-ticket Stop'
    assert b["enforcement_stops"] == 15              # Ticket Only + Ticket and Non-ticket
    assert b["no_enforcement_share"] == pytest.approx(30 / 45)
    assert b["harbor_excluded"] == 8                 # reported, not counted
    # Breakdown is the three street counts (Harbor lives in harbor_excluded).
    assert b["stop_type_breakdown"] == {
        "Non-ticket Stop": 30, "Ticket Only": 10, "Ticket and Non-ticket Stop": 5,
    }
    assert "Harbor" not in b["stop_type_breakdown"]

    nb = r["non_black_known_race"]
    assert nb["street_total"] == 20                  # Unknown-race and Harbor excluded
    assert nb["enforcement_stops"] == 10
    assert nb["no_enforcement_share"] == pytest.approx(0.50)
    assert nb["harbor_excluded"] == 3


def test_harbor_exclusion_count_is_reported_in_exclusions_block():
    cells = group_stop_rows(
        _rows(5, BLACK, "Non-ticket Stop", 30)
        + _rows(5, BLACK, "Harbor", 8)
        + _rows(5, "White", "Harbor", 3)
        + _rows(5, UNKNOWN_RACE, "Harbor", 4)   # unknown-race Harbor: not a street-metric row
        + _rows(None, BLACK, "Harbor", 9)        # unmapped Harbor: not a per-district row
    )
    excl = outcome_split_exclusions(cells)
    # Harbor excluded from the metric = known-race, mapped-district Harbor (8 + 3).
    assert excl["harbor_stops_excluded"] == 11
    assert excl["harbor_stops_excluded_pct"] == pytest.approx(11 / 54 * 100, rel=1e-3)
    # Reported alongside the shared exclusions, same style.
    assert "unknown_race_stops" in excl and "unmapped_district_stops" in excl


def test_outcome_split_no_enforcement_matches_exact_stop_type_string():
    # A near-miss STOP_TYPE string must NOT be counted as no-enforcement, and
    # (being a non-street type) must not inflate the street denominator either.
    cells = group_stop_rows(
        _rows(2, BLACK, NO_TICKET_STOP_TYPE, 10) + _rows(2, BLACK, "non-ticket stop", 90)
    )
    r = outcome_split_by_district(cells, {})[0]
    assert r["black"]["no_enforcement_stops"] == 10  # only the exact label
    assert r["black"]["street_total"] == 10          # the garbage label is not a street type


# ── 5. Time-of-day filter (additive; SQL construction is pure/testable) ────────
def test_time_filter_none_is_unchanged_default():
    # No time window -> no extra WHERE, so the pooled all-hours path is untouched.
    assert _time_filter_sql(None) == ""
    sql = _cells_sql(None)
    assert "WHERE ingest_batch = $1 GROUP BY" in sql
    assert "EXTRACT" not in sql


def test_time_filter_night_uses_local_tz_and_null_guard():
    frag = _time_filter_sql("night")
    assert "America/New_York" in frag              # DC local, not raw UTC hour
    assert "stop_datetime IS NOT NULL" in frag     # null-time rows fall out
    assert " IN (22, 23, 0, 1, 2, 3, 4)" in frag   # 22:00-04:59
    assert set(NIGHT_HOURS) == {22, 23, 0, 1, 2, 3, 4}


def test_time_filter_day_is_complement_of_night():
    assert "NOT IN (22, 23, 0, 1, 2, 3, 4)" in _time_filter_sql("day")
    assert "stop_datetime IS NOT NULL" in _time_filter_sql("day")


def test_time_filter_rejects_bad_window():
    with pytest.raises(ValueError):
        _time_filter_sql("evening")
