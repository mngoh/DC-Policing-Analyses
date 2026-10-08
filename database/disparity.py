"""Disparity engine — the core accountability metric (computation/query only).

Joins the two persisted datasets to produce, per MPD police district:
  * a racial REPRESENTATION INDEX for stops (headline), and
  * a preliminary stop-OUTCOME split by race (secondary),
plus a citywide summary. This module is NOT a surface: no routes, no pages, no
PDF. It exposes clean functions that a later surface layer will call. It does not
touch crime_index.py, the report modules, or the frontend.

Numerator  : ``stops``               (person-level rows; database/stops.py)
Denominator: ``district_population`` + ``citywide_population``  (database/district_population.py)

Definitions are the product — read them exactly:

GRAIN — person-level. ``stops`` rows are per PERSON, not per stop-event
(multi-person stops repeat CCN). We count person-records and pair them against
ACS *person* population. We do NOT collapse to distinct stop-events. Every result
is labelled "stops of persons".

REPRESENTATION INDEX — two baselines, both per district, both exposed:
    Black stop share = Black share of that district's known-race stops.
  * PRIMARY (per district)  ``local_representation_index``
        = Black stop share ÷ that district's OWN Black population share
    Answers "relative to who lives HERE." Empirically this is where the signal
    lives in majority-non-Black districts (e.g. 2D ~6.7x on an 8% Black pop) and
    flattens to ~1.0 in near-single-race districts (6D/7D).
  * SECONDARY (per district) ``representation_index``  (UNCHANGED)
        = Black stop share ÷ the Black CITYWIDE population share (~43.3%)
    Answers "relative to being Black anywhere in DC." Loud in majority-Black
    districts (7D ~2.2x). Kept for comparison; this is also the CITYWIDE SUMMARY
    baseline (``citywide_representation``), which is unchanged.
The two baselines have opposite shapes on purpose. The raw components (Black stop
count, known-race stop total, both baselines) ride beside the indices — never a
black box. The local ratio in a low-Black district rests on a small population
denominator, so ``black_stops`` (the numerator N) is exposed for caveating.

  Baseline choice (FLAGGED, concurred): population baselines are the headline —
  legible, honest, what the denominator was built for. A stop-neutral /
  expected-stops baseline is a more objection-resistant refinement; it is a
  DOCUMENTED FUTURE METRIC (see ``FUTURE_METRIC_STOP_NEUTRAL``), not built here.

EXCLUSIONS — loud, never silent. Two categories are excluded from the index
denominator AND reported explicitly (see ``summarize_exclusions`` and the
``exclusions`` block every entry point returns):
  * ETHNICITY 'Unknown' (and any SQL NULL) — not a known race; excluded from the
    known-race denominator, counted and reported.
  * unmapped district (STOP_DISTRICT 'NA'/null -> ``district IS NULL``) — excluded
    from any per-district metric, counted and reported.

OUTCOME SPLIT (secondary, PRELIMINARY) — per district, by race: the no-enforcement
share of STREET stops. STOP_TYPE ONLY — the arrest/ticket/warning ``*_raw``
columns are garbage-encoded and are neither read nor imported here.

  STOP_TYPE has FOUR live values: 'Non-ticket Stop', 'Ticket Only',
  'Ticket and Non-ticket Stop', 'Harbor'. Harbor stops are marine-unit stops — a
  different population from street stops — and are EXCLUDED from the split
  entirely (neither numerator nor denominator), reported loudly via
  ``harbor_stops_excluded`` in the exclusions block. Among the remaining STREET
  stops: no-enforcement = 'Non-ticket Stop'; enforcement = 'Ticket Only' +
  'Ticket and Non-ticket Stop' (a ticket was issued in both). The metric is
  no-enforcement share = 'Non-ticket Stop' / (all street stops in the group),
  computed Black vs non-Black-known-race per district. The full composition (the
  three street counts + the excluded Harbor count) is returned so the surface can
  show the makeup, not just the ratio, and every outcome result carries
  ``is_preliminary: True``.

CURRENT-BATCH SCOPING (correctness, load-bearing). ``stops`` stores a full
snapshot per ingest run (snapshot-per-release batch versioning). Querying all
rows would sum across snapshots and multiply the counts. Every query here is
scoped to a single batch — by default the latest FINISHED batch for the layer
(``ingest_batch = MAX(batch_id) WHERE source_layer=41 AND finished_at NOT NULL``).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

# ── Definitional constants ────────────────────────────────────────────────────
CURRENT_SOURCE_LAYER = 41
GRAIN = "stops of persons"

BLACK = "Black"                       # ETHNICITY value; aligns with ACS "Black alone" (DP05_0038)
UNKNOWN_RACE = "Unknown"             # literal missing-race label; layer 45 (2019-2022)
                                     # ALSO has ~1.8k true SQL NULLs — both are
                                     # not-known-race via _is_known_race and land
                                     # in the exclusions block, never dropped.
NO_TICKET_STOP_TYPE = "Non-ticket Stop"
HARBOR_STOP_TYPE = "Harbor"          # marine-unit stops — excluded from the outcome split

# The four live STOP_TYPE values. Harbor is a marine-unit category (a different
# population from street stops), excluded from the outcome split entirely. The
# three STREET types are what the outcome metric is computed over; of those, the
# two ENFORCEMENT types mean a ticket was issued.
STOP_TYPE_VALUES = (
    "Non-ticket Stop",
    "Ticket Only",
    "Ticket and Non-ticket Stop",
    "Harbor",
)
STREET_STOP_TYPES = ("Non-ticket Stop", "Ticket Only", "Ticket and Non-ticket Stop")
ENFORCEMENT_STOP_TYPES = ("Ticket Only", "Ticket and Non-ticket Stop")

OUTCOME_DEFINITION = (
    "no-enforcement share = count(STOP_TYPE = 'Non-ticket Stop') / count(street "
    "stops in group). Harbor (marine-unit) stops are EXCLUDED from both numerator "
    "and denominator and reported separately. Of the remaining street stops, "
    "'Ticket Only' and 'Ticket and Non-ticket Stop' are enforcement (a ticket was "
    "issued); 'Non-ticket Stop' is no-enforcement. Preliminary: rests on STOP_TYPE "
    "alone, not a full enforcement measure. The arrest/ticket/warning raw columns "
    "are garbage-encoded and NOT used."
)

FUTURE_METRIC_STOP_NEUTRAL = (
    "Future refinement (not built): a stop-neutral / expected-stops baseline "
    "(observed vs expected Black stops given each district's stop volume) is more "
    "objection-resistant than the population baseline used for the headline index."
)


# ── Small helpers ─────────────────────────────────────────────────────────────
def _is_known_race(ethnicity: Optional[str]) -> bool:
    """Known race = a real category that is neither the 'Unknown' label nor SQL NULL."""
    return ethnicity is not None and ethnicity != UNKNOWN_RACE


def _pct(part: int, whole: int) -> Optional[float]:
    return round(100 * part / whole, 2) if whole else None


def _share(part: int, whole: int) -> Optional[float]:
    return part / whole if whole else None


# ── Pure aggregation over "cells" ─────────────────────────────────────────────
# A "cell" is one grouped count: {district, ethnicity, stop_type, n}. The SQL does
# a plain GROUP BY into cells (no definitional logic); ALL meaning-bearing logic
# lives in these pure functions so it is unit-tested on fixtures. group_stop_rows
# builds the same cells from raw person-rows (used by the tests) so the SQL and the
# tested path bucket identically.
def group_stop_rows(rows: list[dict]) -> list[dict]:
    """Group raw person-rows (keys: district, ethnicity, stop_type) into cells."""
    counter: Counter = Counter(
        (r.get("district"), r.get("ethnicity"), r.get("stop_type")) for r in rows
    )
    return [
        {"district": d, "ethnicity": e, "stop_type": s, "n": n}
        for (d, e, s), n in counter.items()
    ]


def summarize_exclusions(cells: list[dict]) -> dict:
    """The loud excluded-summary attached to every index result.

    Reports the two excluded categories (unknown-race, unmapped-district) and the
    share of ALL stops that the per-district known-race figures rest on.
    """
    total = sum(c["n"] for c in cells)
    unknown_race = sum(c["n"] for c in cells if not _is_known_race(c["ethnicity"]))
    unmapped_district = sum(c["n"] for c in cells if c["district"] is None)
    mapped_known = sum(
        c["n"] for c in cells
        if c["district"] is not None and _is_known_race(c["ethnicity"])
    )
    return {
        "total_stops": total,
        "unknown_race_stops": unknown_race,
        "unknown_race_pct": _pct(unknown_race, total),
        "unmapped_district_stops": unmapped_district,
        "unmapped_district_pct": _pct(unmapped_district, total),
        # % of all stops the reported per-district known-race figures rest on.
        "reported_coverage_pct": _pct(mapped_known, total),
        "grain": GRAIN,
    }


def representation_by_district(
    cells: list[dict],
    citywide_black_pop_share: float,
    district_meta: Optional[dict[int, dict]] = None,
) -> list[dict]:
    """Per-district representation index — LOCAL baseline as the primary metric.

    Two baselines, both exposed per district:
      * PRIMARY   ``local_representation_index``   = Black stop share ÷ that
        district's OWN Black population share (district_population). Answers
        "relative to who lives HERE, how over-represented are Black people in
        this district's stops?" — loud in majority-non-Black districts.
      * SECONDARY ``representation_index``         = Black stop share ÷ the
        CITYWIDE Black population share (unchanged). Answers "relative to being
        Black anywhere in DC" — loud in majority-Black districts.

    The local ratio in a low-Black district rides on a small population
    denominator, so ``black_stops`` (the N the numerator rests on) is exposed for
    caveating thin bases.

    ``citywide_black_pop_share`` is a fraction 0-1 (from citywide_population).
    ``district_meta`` maps district -> {police_district, black_pop_share}; the
    district's own ``black_pop_share`` is the LOCAL baseline.
    """
    district_meta = district_meta or {}
    per: dict[int, dict] = {}
    for c in cells:
        d = c["district"]
        if d is None:          # unmapped -> excluded from any per-district metric
            continue
        agg = per.setdefault(d, {"black": 0, "known": 0, "unknown": 0})
        if _is_known_race(c["ethnicity"]):
            agg["known"] += c["n"]
            if c["ethnicity"] == BLACK:
                agg["black"] += c["n"]
        else:
            agg["unknown"] += c["n"]

    rows: list[dict] = []
    for d in sorted(per):
        agg = per[d]
        share = _share(agg["black"], agg["known"])
        meta = district_meta.get(d, {})
        local_base = meta.get("black_pop_share")   # district's OWN Black pop share
        local_index = (share / local_base
                       if share is not None and local_base else None)
        citywide_index = (share / citywide_black_pop_share
                          if share is not None and citywide_black_pop_share else None)
        rows.append({
            "district": d,
            "police_district": meta.get("police_district"),
            # Raw components — the index is never a black box.
            "black_stops": agg["black"],             # the N the local ratio rests on
            "known_race_stops": agg["known"],
            "black_stop_share": share,
            # PRIMARY: local baseline (this district's own Black population share).
            "local_black_pop_share": local_base,
            "local_representation_index": local_index,
            # SECONDARY: citywide baseline, retained unchanged for comparison.
            "citywide_black_pop_share": citywide_black_pop_share,
            "representation_index": citywide_index,
            # Context alias for the local baseline (kept for back-compat).
            "district_black_pop_share": local_base,
            # Per-district exclusion (excluded from this district's denominator).
            "unknown_race_stops": agg["unknown"],
            "district_total_stops": agg["known"] + agg["unknown"],
        })
    return rows


def citywide_representation(cells: list[dict], citywide_black_pop_share: float) -> dict:
    """Citywide (all districts, incl. unmapped) Black stop share and index."""
    black = sum(c["n"] for c in cells if c["ethnicity"] == BLACK)
    known = sum(c["n"] for c in cells if _is_known_race(c["ethnicity"]))
    share = _share(black, known)
    index = (share / citywide_black_pop_share
             if share is not None and citywide_black_pop_share else None)
    return {
        "black_stops": black,
        "known_race_stops": known,
        "black_stop_share": share,
        "citywide_black_pop_share": citywide_black_pop_share,
        "representation_index": index,
        "grain": GRAIN,
    }


def _outcome_group(stop_type_counts: dict[str, int]) -> dict:
    """no-enforcement share for one (district, race-group).

    Harbor stops are excluded from both numerator and denominator (reported as
    ``harbor_excluded``). The denominator is street stops only; of those, the two
    enforcement types mean a ticket was issued.
    """
    no_enforcement = stop_type_counts.get(NO_TICKET_STOP_TYPE, 0)
    enforcement = sum(stop_type_counts.get(k, 0) for k in ENFORCEMENT_STOP_TYPES)
    street_total = sum(stop_type_counts.get(k, 0) for k in STREET_STOP_TYPES)
    harbor = stop_type_counts.get(HARBOR_STOP_TYPE, 0)
    # Full composition: the three street counts, exposed so the ratio is auditable.
    breakdown = {k: stop_type_counts.get(k, 0) for k in STREET_STOP_TYPES}
    return {
        "no_enforcement_stops": no_enforcement,
        "enforcement_stops": enforcement,
        "street_total": street_total,                 # the denominator (Harbor excluded)
        "no_enforcement_share": _share(no_enforcement, street_total),
        "harbor_excluded": harbor,                    # excluded, reported loudly
        "stop_type_breakdown": breakdown,
    }


def outcome_split_by_district(
    cells: list[dict],
    district_meta: Optional[dict[int, dict]] = None,
) -> list[dict]:
    """Preliminary stop-outcome split: Black vs non-Black-known-race, per district.

    Unknown-race and unmapped-district rows are excluded (and reported via the
    caller's exclusions block). Uses STOP_TYPE only.
    """
    district_meta = district_meta or {}
    per: dict[int, dict] = {}
    for c in cells:
        d = c["district"]
        if d is None or not _is_known_race(c["ethnicity"]):
            continue
        group = "black" if c["ethnicity"] == BLACK else "non_black_known_race"
        agg = per.setdefault(d, {"black": {}, "non_black_known_race": {}})
        st = c["stop_type"]
        agg[group][st] = agg[group].get(st, 0) + c["n"]

    rows: list[dict] = []
    for d in sorted(per):
        agg = per[d]
        rows.append({
            "district": d,
            "police_district": district_meta.get(d, {}).get("police_district"),
            "black": _outcome_group(agg["black"]),
            "non_black_known_race": _outcome_group(agg["non_black_known_race"]),
            "is_preliminary": True,
            "definition": OUTCOME_DEFINITION,
        })
    return rows


def outcome_split_exclusions(cells: list[dict]) -> dict:
    """Exclusions block for the outcome split: the shared unknown-race / unmapped
    summary PLUS the Harbor stops excluded from the metric (known-race, mapped
    street population), reported in the same style — never silently dropped."""
    base = summarize_exclusions(cells)
    harbor = sum(
        c["n"] for c in cells
        if c["district"] is not None and _is_known_race(c["ethnicity"])
        and c["stop_type"] == HARBOR_STOP_TYPE
    )
    base["harbor_stops_excluded"] = harbor
    base["harbor_stops_excluded_pct"] = _pct(harbor, base["total_stops"])
    return base


# ── Time-of-day windows (additive) ────────────────────────────────────────────
# night = 22:00–04:59, day = everything else. The hour is taken in DC LOCAL time
# (America/New_York), because "10pm–5am" is a wall-clock concept; stop_datetime is
# stored as TIMESTAMPTZ (UTC), so a bare UTC hour would shift the window ~4-5h.
# time_window=None is the pooled all-hours path (default, unchanged). Rows with a
# NULL/unparseable stop_datetime have no hour and fall out of BOTH day and night
# (report the null-time share via time_coverage()).
TIME_TZ = "America/New_York"
NIGHT_HOURS = (22, 23, 0, 1, 2, 3, 4)
TIME_WINDOWS = ("day", "night")
_NIGHT_LIST = ", ".join(str(h) for h in NIGHT_HOURS)
_NIGHT_HOUR_EXPR = f"EXTRACT(HOUR FROM stop_datetime AT TIME ZONE '{TIME_TZ}')"


def _time_filter_sql(time_window: Optional[str]) -> str:
    """Extra WHERE fragment for a time window (constants only; no user input)."""
    if time_window is None:
        return ""
    if time_window not in TIME_WINDOWS:
        raise ValueError(f"time_window must be None, 'day', or 'night'; got {time_window!r}")
    op = "IN" if time_window == "night" else "NOT IN"
    return (f" AND stop_datetime IS NOT NULL "
            f"AND {_NIGHT_HOUR_EXPR} {op} ({_NIGHT_LIST})")


def _cells_sql(time_window: Optional[str]) -> str:
    return (
        "SELECT district, ethnicity, stop_type, COUNT(*) AS n FROM stops "
        f"WHERE ingest_batch = $1{_time_filter_sql(time_window)} "
        "GROUP BY district, ethnicity, stop_type"
    )


# ── Search & seizure outcome test (additive) ──────────────────────────────────
# The strongest bias statistic this data supports (the "outcome test", after the
# Stanford Open Policing Project): compare not just how often each group is
# SEARCHED, but how often searches of each group actually FIND something. If one
# group is searched more but searches of that group yield contraband less often,
# searches of that group rest on thinner suspicion — a conclusion no population
# baseline is needed for, because the denominator is the searches themselves.
#
# Source shape (verified live on both layers 2026-07-12):
#   * Four person-search FLAGS (consent / probable-cause / warrant / protective
#     pat-down): clean numeric booleans, no NULLs. These define who was searched.
#   * Three seizure-outcome TEXT fields (consent, warrant, pat-down): messy,
#     multi-value, era-varying vocabulary — parsed by parse_seizure() below.
#   * There is NO seizure-outcome field for probable-cause person searches in
#     the source. That type gets search rates only and is EXCLUDED from hit
#     rates; the exclusion is stated in the definition and the surface.
#   * Rows where a flag says searched but the outcome text is NULL/'NA' are
#     "outcome unrecorded": excluded from the hit-rate denominator, counted and
#     reported — never guessed to a hit or a miss.
SEARCH_TYPES = ("consent", "prob_cause", "warrant", "pat_down")
HIT_RATE_TYPES = ("consent", "warrant", "pat_down")   # prob_cause has no outcome field

OUTCOME_TEST_DEFINITION = (
    "search rate = persons searched (flag) / stops in group; hit rate = searches "
    "where something was seized / searches with a RECORDED outcome. Outcomes are "
    "recorded only for consent, warrant, and pat-down searches (the source has no "
    "outcome field for probable-cause searches, so they are in search rates but "
    "not hit rates). 'None' outcomes count as recorded misses; NULL/'NA' outcomes "
    "are unrecorded — excluded from the hit-rate denominator and reported. "
    "Groups: Black vs non-Black known race; unknown-race stops excluded and "
    "reported."
)

_SEIZED_SKIP_TOKENS = {"na"}       # not-applicable/unrecorded marker, not a miss
_SEIZED_MISS_TOKENS = {"none"}     # recorded "nothing seized"


def parse_seizure(text: Optional[str]) -> Optional[bool]:
    """One seizure-outcome text -> True (something seized), False (recorded
    'None'), or None (no outcome recorded).

    Values are multi-item lists with era-varying separators/vocabulary
    ('Gun ;', 'Alcohol, None', 'None ;', 'NA', 'Drugs/narcotics'). Tokenize on
    ';' and ',', drop empty/'NA' tokens; any remaining non-'None' token means
    something was seized ('Alcohol, None' -> seized). Only-'None' -> recorded
    miss. Nothing left ('NA', '') -> unrecorded.
    """
    if text is None:
        return None
    tokens = [t.strip().lower() for part in text.split(";") for t in part.split(",")]
    real = [t for t in tokens if t and t not in _SEIZED_SKIP_TOKENS]
    if not real:
        return None
    return any(t not in _SEIZED_MISS_TOKENS for t in real)


def _race_group(ethnicity: Optional[str]) -> Optional[str]:
    """Black / non_black_known / None (excluded: Unknown label or SQL NULL)."""
    if not _is_known_race(ethnicity):
        return None
    return "black" if ethnicity == BLACK else "non_black_known_race"


def search_rates_by_district(cells: list[dict]) -> list[dict]:
    """Per-district search rates, Black vs non-Black known race.

    ``cells``: {district, ethnicity, n, consent, prob_cause, warrant, pat_down,
    any_search, any_search_or_pat_down} — counts of stopped persons and of
    searched persons per flag (any_search = consent|prob_cause|warrant).
    Unknown-race and unmapped-district cells are excluded here and reported via
    the caller's exclusions block.
    """
    per: dict[int, dict] = {}
    fields = SEARCH_TYPES + ("any_search", "any_search_or_pat_down")
    for c in cells:
        d, g = c.get("district"), _race_group(c.get("ethnicity"))
        if d is None or g is None:
            continue
        agg = per.setdefault(d, {
            "black": dict.fromkeys(("stops",) + fields, 0),
            "non_black_known_race": dict.fromkeys(("stops",) + fields, 0),
        })
        agg[g]["stops"] += c["n"]
        for f in fields:
            agg[g][f] += c.get(f) or 0

    rows: list[dict] = []
    for d in sorted(per):
        row: dict = {"district": d}
        for g in ("black", "non_black_known_race"):
            a = per[d][g]
            row[g] = {
                **a,
                "search_rate": _share(a["any_search"], a["stops"]),
                "pat_down_rate": _share(a["pat_down"], a["stops"]),
            }
        b, nb = row["black"], row["non_black_known_race"]
        row["pat_down_rate_ratio"] = (
            b["pat_down_rate"] / nb["pat_down_rate"]
            if b["pat_down_rate"] is not None and nb["pat_down_rate"] else None)
        rows.append(row)
    return rows


def hit_rates_citywide(hit_cells: list[dict]) -> dict:
    """Citywide hit rates by race group, per search type and combined.

    ``hit_cells``: {ethnicity, search_type, outcome, n} where outcome is
    True/False/None from parse_seizure. Includes unmapped-district rows (a
    search's outcome does not depend on mapping the district); known race only.
    """
    groups = {
        g: {t: {"searched": 0, "seized": 0, "recorded": 0, "unrecorded": 0}
            for t in HIT_RATE_TYPES + ("combined",)}
        for g in ("black", "non_black_known_race")
    }
    for c in hit_cells:
        g = _race_group(c.get("ethnicity"))
        t = c.get("search_type")
        if g is None or t not in HIT_RATE_TYPES:
            continue
        for bucket in (groups[g][t], groups[g]["combined"]):
            bucket["searched"] += c["n"]
            if c.get("outcome") is None:
                bucket["unrecorded"] += c["n"]
            else:
                bucket["recorded"] += c["n"]
                if c["outcome"]:
                    bucket["seized"] += c["n"]
    for g in groups.values():
        for t in g.values():
            t["hit_rate"] = _share(t["seized"], t["recorded"])
    return groups


def outcome_test_direction(black: dict, non_black: dict) -> dict:
    """Direction of the hit-rate gap, gated on statistical significance.

    A raw threshold would call 13.9% vs 12.5% 'higher', but with ~1k searches
    in the smaller group that gap is ~1.3 standard errors — noise. The most
    contested statistic on the site must never assert a direction the data
    cannot support, so: two-proportion z-test (pooled), two-sided, alpha .05.
    Non-significant gaps are 'about the same' with significant=False; the
    surface says 'statistically indistinguishable', never a direction.

    Inputs are the 'combined' hit-rate buckets: {seized, recorded, hit_rate}.
    """
    n1, n2 = black.get("recorded") or 0, non_black.get("recorded") or 0
    if not n1 or not n2:
        return {"direction": "not comparable", "z": None, "significant": False}
    p1, p2 = black["seized"] / n1, non_black["seized"] / n2
    pooled = (black["seized"] + non_black["seized"]) / (n1 + n2)
    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se else 0.0
    if abs(z) < 1.96:
        return {"direction": "about the same", "z": round(z, 2), "significant": False}
    return {"direction": "lower" if z < 0 else "higher",
            "z": round(z, 2), "significant": True}


# ── Yearly trend across layers (additive) ─────────────────────────────────────
# The trend spans BOTH source layers — 45 (2019-07-22 to 2022) and 41 (2023-
# present) — one latest-finished batch per layer, so it is the same snapshot
# discipline as the single-window metrics, just across two snapshots. Years are
# UTC-derived (stops.year, mirroring crimes.year); the handful of NYE stops that
# cross midnight into the next UTC year stay where the shared convention puts
# them. Rows with a NULL stop_datetime have no year and are excluded from every
# per-year figure — counted and reported via ``null_year_stops``, never dropped
# silently. The trend is ALL-HOURS only (no day/night variant).
TREND_LAYERS = (45, 41)              # chronological
TREND_BASELINE_NOTE = (
    "One ACS population baseline (see acs_vintage) is applied to every year of "
    "the trend. Population shares move slowly, but the early years lean on a "
    "denominator measured later; year-over-year changes in the index reflect "
    "changes in stops, not population."
)


def group_year_rows(rows: list[dict]) -> list[dict]:
    """Group raw person-rows (keys: year, district, ethnicity) into year-cells.

    The test-path twin of the trend SQL (same role group_stop_rows plays for the
    single-window cells) so the SQL and the tested path bucket identically.
    """
    counter: Counter = Counter(
        (r.get("year"), r.get("district"), r.get("ethnicity")) for r in rows
    )
    return [
        {"year": y, "district": d, "ethnicity": e, "n": n}
        for (y, d, e), n in counter.items()
    ]


def trend_by_year(
    cells: list[dict],
    citywide_black_pop_share: float,
    district_meta: Optional[dict[int, dict]] = None,
) -> list[dict]:
    """Per-year representation results from year-cells, ascending year order.

    Reuses the SAME pure functions as the single-window path (citywide index,
    per-district local/citywide indices, exclusions summary) so a trend point is
    definitionally identical to the headline metric computed on that year alone.
    Cells with a NULL year are skipped here; the caller reports them.
    """
    by_year: dict[int, list[dict]] = {}
    for c in cells:
        if c.get("year") is None:
            continue
        by_year.setdefault(c["year"], []).append(c)
    out: list[dict] = []
    for y in sorted(by_year):
        ycells = by_year[y]
        out.append({
            "year": y,
            "citywide": citywide_representation(ycells, citywide_black_pop_share),
            "districts": representation_by_district(
                ycells, citywide_black_pop_share, district_meta),
            "exclusions": summarize_exclusions(ycells),
        })
    return out


# ── DB access (thin: GROUP BY -> cells, then the pure functions above) ─────────
_LATEST_BATCH_SQL = """
SELECT MAX(batch_id) FROM stop_ingest_batches
WHERE source_layer = $1 AND finished_at IS NOT NULL
"""


async def resolve_current_batch(conn, source_layer: int = CURRENT_SOURCE_LAYER) -> Optional[int]:
    """The latest FINISHED ingest batch for the layer ("current" snapshot)."""
    return await conn.fetchval(_LATEST_BATCH_SQL, source_layer)


async def time_coverage(conn, *, source_layer: int = CURRENT_SOURCE_LAYER,
                        batch_id: Optional[int] = None) -> dict:
    """Data-quality check for a time split: how many stop rows have NO usable time.

    Rows with a NULL stop_datetime cannot be placed in day or night, so a day/night
    split rests on (total - null_time) rows. Reports the null-time count/%% plus the
    day/night breakdown of the timed rows, all for the current batch.
    """
    batch = batch_id if batch_id is not None else await resolve_current_batch(conn, source_layer)
    if batch is None:
        raise RuntimeError(
            f"no finished stop-ingest batch for source_layer={source_layer} — run "
            "scripts/ingest_stops.py first."
        )
    row = await conn.fetchrow(
        f"""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE stop_datetime IS NULL) AS null_time,
               COUNT(*) FILTER (WHERE stop_datetime IS NOT NULL
                                AND {_NIGHT_HOUR_EXPR} IN ({_NIGHT_LIST})) AS night,
               COUNT(*) FILTER (WHERE stop_datetime IS NOT NULL
                                AND {_NIGHT_HOUR_EXPR} NOT IN ({_NIGHT_LIST})) AS day
        FROM stops WHERE ingest_batch = $1
        """,
        batch,
    )
    total = row["total"] or 0
    null_time = row["null_time"] or 0
    return {
        "ingest_batch": batch,
        "total_stops": total,
        "null_time_stops": null_time,
        "null_time_pct": round(100 * null_time / total, 2) if total else None,
        "day_stops": row["day"] or 0,
        "night_stops": row["night"] or 0,
        "timezone": TIME_TZ,
    }


async def _fetch_cells(conn, batch_id: int, time_window: Optional[str] = None) -> list[dict]:
    rows = await conn.fetch(_cells_sql(time_window), batch_id)
    return [
        {"district": r["district"], "ethnicity": r["ethnicity"],
         "stop_type": r["stop_type"], "n": r["n"]}
        for r in rows
    ]


async def _fetch_citywide_baseline(conn, scope: str = "dc") -> tuple[float, dict]:
    row = await conn.fetchrow(
        "SELECT total_population, black_population, pct_black, acs_vintage "
        "FROM citywide_population WHERE scope = $1", scope,
    )
    if row is None or not row["total_population"]:
        raise RuntimeError(
            "citywide_population baseline not loaded — run "
            "scripts/load_district_population.py before computing the index."
        )
    # Use the persisted counts for the fraction (exposed in output; the ~43.3%
    # pct_black column is the rounded display of this same number).
    share = row["black_population"] / row["total_population"]
    info = {"total_population": row["total_population"],
            "black_population": row["black_population"],
            "pct_black_display": float(row["pct_black"]),
            "acs_vintage": row["acs_vintage"]}
    return share, info


async def _fetch_district_meta(conn) -> dict[int, dict]:
    rows = await conn.fetch(
        "SELECT district, police_district, total_population, black_population "
        "FROM district_population"
    )
    return {
        r["district"]: {
            "police_district": r["police_district"],
            "black_pop_share": (r["black_population"] / r["total_population"]
                                if r["total_population"] else None),
        }
        for r in rows
    }


async def _prepare(conn, source_layer: int, batch_id: Optional[int],
                   time_window: Optional[str] = None):
    batch = batch_id if batch_id is not None else await resolve_current_batch(conn, source_layer)
    if batch is None:
        raise RuntimeError(
            f"no finished stop-ingest batch for source_layer={source_layer} — run "
            "scripts/ingest_stops.py before computing the index."
        )
    cells = await _fetch_cells(conn, batch, time_window)
    baseline, baseline_info = await _fetch_citywide_baseline(conn)
    district_meta = await _fetch_district_meta(conn)
    meta = {
        "grain": GRAIN,
        "baseline": "citywide",
        "source_layer": source_layer,
        "ingest_batch": batch,
        "time_window": time_window or "all",
        "citywide_black_pop_share": baseline,
        "baseline_source": baseline_info,
    }
    return cells, baseline, district_meta, meta


async def district_representation_index(
    conn, *, source_layer: int = CURRENT_SOURCE_LAYER, batch_id: Optional[int] = None,
    time_window: Optional[str] = None,
) -> dict:
    """Headline: per-district representation index + raw components + exclusions.

    ``time_window``: None (all hours, default), 'day', or 'night' (22:00-04:59 DC
    local). The denominator population baseline is unchanged; only the stop
    numerator is restricted to the window.
    """
    cells, baseline, district_meta, meta = await _prepare(conn, source_layer, batch_id, time_window)
    return {
        "meta": meta,
        "districts": representation_by_district(cells, baseline, district_meta),
        "exclusions": summarize_exclusions(cells),
        "future_metric": FUTURE_METRIC_STOP_NEUTRAL,
    }


async def citywide_disparity_summary(
    conn, *, source_layer: int = CURRENT_SOURCE_LAYER, batch_id: Optional[int] = None,
    time_window: Optional[str] = None,
) -> dict:
    """Citywide Black stop share and index (sanity anchor for the per-district set)."""
    cells, baseline, _district_meta, meta = await _prepare(conn, source_layer, batch_id, time_window)
    return {
        "meta": meta,
        "citywide": citywide_representation(cells, baseline),
        "exclusions": summarize_exclusions(cells),
    }


async def stop_outcome_split(
    conn, *, source_layer: int = CURRENT_SOURCE_LAYER, batch_id: Optional[int] = None,
    time_window: Optional[str] = None,
) -> dict:
    """Preliminary per-district no-ticket split by race (STOP_TYPE only)."""
    cells, _baseline, district_meta, meta = await _prepare(conn, source_layer, batch_id, time_window)
    return {
        "meta": {**meta, "is_preliminary": True, "definition": OUTCOME_DEFINITION},
        "districts": outcome_split_by_district(cells, district_meta),
        "exclusions": outcome_split_exclusions(cells),
    }


# ── Yearly trend entry point ──────────────────────────────────────────────────
_TREND_CELLS_SQL = """
SELECT year, district, ethnicity, COUNT(*) AS n FROM stops
WHERE ingest_batch = ANY($1::bigint[])
GROUP BY year, district, ethnicity
"""

_TREND_SPAN_SQL = """
SELECT MIN(stop_datetime) AS min_dt, MAX(stop_datetime) AS max_dt,
       COUNT(*) FILTER (WHERE stop_datetime IS NULL) AS null_year_stops
FROM stops WHERE ingest_batch = ANY($1::bigint[])
"""


async def resolve_trend_batches(conn, layers: tuple = TREND_LAYERS) -> dict[int, int]:
    """Latest FINISHED batch per layer; layers with no finished batch are omitted
    (the caller reports which layers the trend actually rests on)."""
    out: dict[int, int] = {}
    for layer in layers:
        batch = await resolve_current_batch(conn, layer)
        if batch is not None:
            out[layer] = batch
    return out


def _partial_year_labels(min_dt, max_dt) -> dict[int, str]:
    """Data-driven partial-year flags for the trend's edge years.

    The first year is partial when coverage starts after early January (layer 45
    begins 2019-07-22 — the start of NEAR Act collection); the last year is
    partial when coverage stops before late December (a mid-year MPD release).
    Read from the data, never hardcoded, so a future release can't go stale.
    """
    labels: dict[int, str] = {}
    if min_dt is not None and (min_dt.month, min_dt.day) > (1, 7):
        labels[min_dt.year] = f"from {min_dt.strftime('%b')} {min_dt.day}"
    if max_dt is not None and (max_dt.month, max_dt.day) < (12, 24):
        labels[max_dt.year] = f"through {max_dt.strftime('%b')} {max_dt.day}"
    return labels


async def yearly_disparity_trend(conn, *, layers: tuple = TREND_LAYERS) -> dict:
    """Per-year citywide + per-district representation across both stop layers.

    Raises RuntimeError when NO layer has a finished batch (same contract as the
    single-window entry points). With only one layer present the trend covers the
    years that layer holds and ``meta.layers`` says so — the surface decides
    whether that is worth rendering.
    """
    batches = await resolve_trend_batches(conn, layers)
    if not batches:
        raise RuntimeError(
            "no finished stop-ingest batch for any trend layer — run "
            "scripts/ingest_stops.py first."
        )
    batch_ids = list(batches.values())
    rows = await conn.fetch(_TREND_CELLS_SQL, batch_ids)
    cells = [
        {"year": r["year"], "district": r["district"],
         "ethnicity": r["ethnicity"], "n": r["n"]}
        for r in rows
    ]
    baseline, baseline_info = await _fetch_citywide_baseline(conn)
    district_meta = await _fetch_district_meta(conn)
    span = await conn.fetchrow(_TREND_SPAN_SQL, batch_ids)
    years = trend_by_year(cells, baseline, district_meta)
    return {
        "meta": {
            "grain": GRAIN,
            "layers": batches,                    # {source_layer: batch_id} actually used
            "time_window": "all",
            "citywide_black_pop_share": baseline,
            "baseline_source": baseline_info,
            "baseline_note": TREND_BASELINE_NOTE,
        },
        "years": years,
        "coverage": {
            "min_stop_datetime": span["min_dt"],
            "max_stop_datetime": span["max_dt"],
            "null_year_stops": span["null_year_stops"] or 0,
            "partial_years": _partial_year_labels(span["min_dt"], span["max_dt"]),
        },
    }


# ── Outcome-test entry point ──────────────────────────────────────────────────
_SEARCH_CELLS_SQL = """
SELECT district, ethnicity, COUNT(*) AS n,
       COUNT(*) FILTER (WHERE search_consent)    AS consent,
       COUNT(*) FILTER (WHERE search_prob_cause) AS prob_cause,
       COUNT(*) FILTER (WHERE search_warrant)    AS warrant,
       COUNT(*) FILTER (WHERE pat_down)          AS pat_down,
       COUNT(*) FILTER (WHERE search_consent OR search_prob_cause
                           OR search_warrant)    AS any_search,
       COUNT(*) FILTER (WHERE search_consent OR search_prob_cause
                           OR search_warrant OR pat_down) AS any_search_or_pat_down
FROM stops WHERE ingest_batch = $1
GROUP BY district, ethnicity
"""

# Distinct raw seizure strings are low-cardinality (~40 per type); grouping by
# the raw text keeps parse_seizure in tested Python instead of SQL.
_HIT_CELLS_SQL = """
SELECT ethnicity, 'consent' AS search_type, seized_consent_raw AS seized_raw, COUNT(*) AS n
FROM stops WHERE ingest_batch = $1 AND search_consent GROUP BY ethnicity, seized_consent_raw
UNION ALL
SELECT ethnicity, 'warrant', seized_warrant_raw, COUNT(*)
FROM stops WHERE ingest_batch = $1 AND search_warrant GROUP BY ethnicity, seized_warrant_raw
UNION ALL
SELECT ethnicity, 'pat_down', seized_pat_down_raw, COUNT(*)
FROM stops WHERE ingest_batch = $1 AND pat_down GROUP BY ethnicity, seized_pat_down_raw
"""

_HAS_SEARCH_DATA_SQL = """
SELECT EXISTS (SELECT 1 FROM stops WHERE ingest_batch = $1 AND pat_down IS NOT NULL)
"""


async def search_outcome_test(
    conn, *, source_layer: int = CURRENT_SOURCE_LAYER, batch_id: Optional[int] = None,
) -> dict:
    """Search rates (per district) + hit rates (citywide) by race group.

    Raises RuntimeError when the current batch predates the search fields (all
    NULL) — an ingest re-run is required, and zeros must never masquerade as
    'no searches happened'.
    """
    batch = batch_id if batch_id is not None else await resolve_current_batch(conn, source_layer)
    if batch is None:
        raise RuntimeError(
            f"no finished stop-ingest batch for source_layer={source_layer} — run "
            "scripts/ingest_stops.py before computing the outcome test."
        )
    if not await conn.fetchval(_HAS_SEARCH_DATA_SQL, batch):
        raise RuntimeError(
            f"batch {batch} predates the person-search fields — re-run "
            "scripts/ingest_stops.py so the current snapshot carries them."
        )
    rows = await conn.fetch(_SEARCH_CELLS_SQL, batch)
    cells = [dict(r) for r in rows]
    hit_rows = await conn.fetch(_HIT_CELLS_SQL, batch)
    hit_cells = [
        {"ethnicity": r["ethnicity"], "search_type": r["search_type"],
         "outcome": parse_seizure(r["seized_raw"]), "n": r["n"]}
        for r in hit_rows
    ]
    citywide = hit_rates_citywide(hit_cells)
    direction = outcome_test_direction(
        citywide["black"]["combined"], citywide["non_black_known_race"]["combined"])
    return {
        "meta": {
            "grain": GRAIN,
            "source_layer": source_layer,
            "ingest_batch": batch,
            "definition": OUTCOME_TEST_DEFINITION,
        },
        "districts": search_rates_by_district(cells),
        "citywide_hit_rates": citywide,
        "direction": direction,
        "exclusions": summarize_exclusions(cells),
    }
