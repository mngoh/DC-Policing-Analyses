"""Enforcement vs reported crime: the crime-denominator baseline.

The strongest objection to a residential-population baseline is "police go
where crime is." This module tests that defense directly: if deployment simply
followed crime, each district's share of stops, frisks, and arrests would match
its share of victim-initiated reported crime. The skew (enforcement share
divided by crime share, per district) measures the departure.

THE DENOMINATOR IS VICTIM-INITIATED BY CONSTRUCTION. DC's public incident feed
contains ONLY the nine Part I offense categories (verified live 2026-07-13):
theft, theft from auto, motor vehicle theft, robbery, burglary, assault with a
dangerous weapon, sex abuse, homicide, arson. Officer-generated offenses
(possession, disorderly, loitering, warrant service) are not published in this
dataset at all, so the circularity that plagues crime-baseline studies (police
presence manufacturing the denominator) is structurally excluded here.

What this is NOT: a racial index on its own. Incident records carry no race
(no offender or victim demographics are published), so the racial reading
comes from the JOIN with district composition: the module computes, live, the
stops-per-reported-crime ratio for majority-Black districts against the rest.
Windows are matched per numerator: stops and frisks (2023-2025, the current
stop era) against crimes 2023-2025; arrests (2013-2025) against crimes
2013-2025.

Same contracts as every other engine: loud exclusions (district-unmapped rows
counted, never dropped silently), pure functions for the math, RuntimeError
when data is missing so surfaces degrade instead of erroring.
"""
from __future__ import annotations

from typing import Any, Optional

# The full offense taxonomy of DC's public incident feed. All nine are
# victim-initiated Part I offenses; listed explicitly so a future feed change
# (a tenth category appearing) is caught by the audit rather than absorbed.
VICTIM_OFFENSES = (
    "THEFT/OTHER", "THEFT F/AUTO", "MOTOR VEHICLE THEFT", "ROBBERY",
    "BURGLARY", "ASSAULT W/DANGEROUS WEAPON", "SEX ABUSE", "HOMICIDE", "ARSON",
)
# Robustness slice: the four violent offenses. Theft dominates the full
# denominator and concentrates in commercial corridors; a hostile reader will
# ask what happens when the denominator is violence, so the page shows both.
VIOLENT_OFFENSES = ("ROBBERY", "ASSAULT W/DANGEROUS WEAPON", "SEX ABUSE", "HOMICIDE")

STOPS_WINDOW = (2023, 2025)      # current stop era (layer 41)
ARRESTS_WINDOW = (2013, 2025)    # full arrest record

CRIME_BASELINE_DEFINITION = (
    "skew = a district's share of enforcement (stops, frisks, or arrests) "
    "divided by its share of victim-initiated reported crime, over a matched "
    "window. 1.0 means enforcement lands exactly where reported crime is; "
    "above 1.0 means a district draws more enforcement than its reported "
    "crime accounts for. The denominator is DC's incident feed, which "
    "publishes only the nine Part I victim/property offenses, so "
    "officer-generated offenses cannot inflate it. Reported crime is not "
    "occurred crime: reporting rates vary by neighborhood, which is why this "
    "page sits beside the residential baseline rather than replacing it."
)


# ── Pure math ─────────────────────────────────────────────────────────────────
def shares(counts: dict[int, int]) -> dict[int, Optional[float]]:
    total = sum(counts.values())
    return {d: (n / total if total else None) for d, n in counts.items()}


def skew_table(crime_counts: dict[int, int],
               numerators: dict[str, dict[int, int]],
               district_meta: Optional[dict] = None) -> list[dict]:
    """Per-district: crime share, each numerator's share, and its skew."""
    district_meta = district_meta or {}
    crime_share = shares(crime_counts)
    out = []
    for d in sorted(set(crime_counts) | {d for c in numerators.values() for d in c}):
        row: dict[str, Any] = {
            "district": d,
            "police_district": district_meta.get(d, {}).get("police_district"),
            "black_pop_share": district_meta.get(d, {}).get("black_pop_share"),
            "crimes": crime_counts.get(d, 0),
            "crime_share": crime_share.get(d),
        }
        for name, counts in numerators.items():
            total = sum(counts.values())
            n = counts.get(d, 0)
            share = (n / total) if total else None
            row[name] = n
            row[f"{name}_share"] = share
            cs = crime_share.get(d)
            row[f"{name}_skew"] = (share / cs) if share is not None and cs else None
        out.append(row)
    return out


def group_ratio(crime_counts: dict[int, int], numer_counts: dict[int, int],
                district_meta: dict, threshold: float = 0.5) -> dict:
    """Enforcement per reported crime, majority-Black districts vs the rest.

    ratio_of_ratios = (enforcement/crime in majority-Black districts) divided
    by (enforcement/crime in the rest). 1.0 means enforcement per reported
    crime is even across the two groups; the direction is read, never assumed.
    """
    mb = {d for d, m in district_meta.items() if (m.get("black_pop_share") or 0) >= threshold}
    def rate(ds):
        crimes = sum(n for d, n in crime_counts.items() if d in ds)
        enf = sum(n for d, n in numer_counts.items() if d in ds)
        return (enf / crimes if crimes else None), enf, crimes
    others = set(crime_counts) | set(numer_counts)
    mb_rate, mb_enf, mb_crimes = rate(mb)
    rest_rate, rest_enf, rest_crimes = rate(others - mb)
    ratio = (mb_rate / rest_rate) if mb_rate is not None and rest_rate else None
    return {
        "majority_black_districts": sorted(mb),
        "majority_black": {"per_crime": mb_rate, "enforcement": mb_enf, "crimes": mb_crimes},
        "rest": {"per_crime": rest_rate, "enforcement": rest_enf, "crimes": rest_crimes},
        "ratio": ratio,
    }


def frisk_hits_by_geography(rows: list[dict], majority_black: set[int]) -> dict:
    """The pairing for the frisks-per-crime skew: do frisks in majority-Black
    districts find contraband at a different rate? rows: {district, outcome, n}
    with outcome from parse_seizure (True/False/None). Direction significance-
    gated with the same two-proportion z-test as the outcome test."""
    from database.disparity import outcome_test_direction
    groups = {"majority_black": {"seized": 0, "recorded": 0, "unrecorded": 0},
              "rest": {"seized": 0, "recorded": 0, "unrecorded": 0}}
    for r in rows:
        if r.get("district") is None:
            continue
        g = groups["majority_black" if r["district"] in majority_black else "rest"]
        if r["outcome"] is None:
            g["unrecorded"] += r["n"]
        else:
            g["recorded"] += r["n"]
            if r["outcome"]:
                g["seized"] += r["n"]
    for g in groups.values():
        g["hit_rate"] = g["seized"] / g["recorded"] if g["recorded"] else None
    return {**groups,
            "direction": outcome_test_direction(groups["majority_black"], groups["rest"])}


# ── DB access ─────────────────────────────────────────────────────────────────
_CRIME_COUNTS_SQL = """
SELECT district, COUNT(*) AS n FROM crimes
WHERE year BETWEEN $1 AND $2 AND offense = ANY($3::text[])
GROUP BY district
"""

_STOP_COUNTS_SQL = """
SELECT district, COUNT(*) AS n,
       COUNT(*) FILTER (WHERE pat_down) AS frisks
FROM stops WHERE ingest_batch = $1
GROUP BY district
"""

_ARREST_COUNTS_SQL = """
SELECT district, COUNT(*) AS n FROM arrests
WHERE ingest_batch = $1 AND year BETWEEN $2 AND $3
GROUP BY district
"""

_FRISK_OUTCOME_SQL = """
SELECT district, seized_pat_down_raw AS raw, COUNT(*) AS n FROM stops
WHERE ingest_batch = $1 AND pat_down AND district IS NOT NULL
GROUP BY district, seized_pat_down_raw
"""


async def enforcement_vs_crime(conn) -> dict:
    """Per-district enforcement-vs-crime skews plus the majority-Black group
    ratio, windows matched per numerator."""
    import asyncpg

    from database.arrests import _LATEST_ARREST_BATCH_SQL
    from database.disparity import _fetch_district_meta, resolve_current_batch

    stop_batch = await resolve_current_batch(conn, 41)
    if stop_batch is None:
        raise RuntimeError("no finished stop-ingest batch - run scripts/ingest_stops.py first.")
    try:
        arrest_batch = await conn.fetchval(_LATEST_ARREST_BATCH_SQL)
    except asyncpg.UndefinedTableError:
        arrest_batch = None

    def _dist(v) -> Optional[int]:
        """crimes.district is TEXT ('1'..'7'); stops/arrests are SMALLINT.
        Normalize to int 1-7, anything else is unmapped."""
        try:
            n = int(str(v).strip())
            return n if 1 <= n <= 7 else None
        except (ValueError, TypeError):
            return None

    def by_d(rows, key="n"):
        out: dict[int, int] = {}
        for r in rows:
            d = _dist(r["district"])
            if d is not None:
                out[d] = out.get(d, 0) + r[key]
        return out

    def unmapped(rows, key="n"):
        return sum(r[key] for r in rows if _dist(r["district"]) is None)

    crimes_now = await conn.fetch(_CRIME_COUNTS_SQL, *STOPS_WINDOW, list(VICTIM_OFFENSES))
    violent_now = await conn.fetch(_CRIME_COUNTS_SQL, *STOPS_WINDOW, list(VIOLENT_OFFENSES))
    stop_rows = await conn.fetch(_STOP_COUNTS_SQL, stop_batch)

    numerators = {"stops": by_d(stop_rows), "frisks": by_d(stop_rows, "frisks")}
    exclusions = {
        "crimes_unmapped_district": unmapped(crimes_now),
        "stops_unmapped_district": unmapped(stop_rows),
    }
    district_meta = await _fetch_district_meta(conn)
    table_now = skew_table(by_d(crimes_now), numerators, district_meta)

    # The pairing: hit rates of those frisks, split the same geographic way.
    from database.disparity import parse_seizure
    mb = {d for d, m in district_meta.items() if (m.get("black_pop_share") or 0) >= 0.5}
    hit_rows = await conn.fetch(_FRISK_OUTCOME_SQL, stop_batch)
    frisk_hits = frisk_hits_by_geography(
        [{"district": r["district"], "outcome": parse_seizure(r["raw"]), "n": r["n"]}
         for r in hit_rows], mb)

    arrests_block = None
    if arrest_batch is not None:
        crimes_long = await conn.fetch(_CRIME_COUNTS_SQL, *ARRESTS_WINDOW, list(VICTIM_OFFENSES))
        violent_long = await conn.fetch(_CRIME_COUNTS_SQL, *ARRESTS_WINDOW, list(VIOLENT_OFFENSES))
        arrest_rows = await conn.fetch(_ARREST_COUNTS_SQL, arrest_batch, *ARRESTS_WINDOW)
        exclusions["arrests_unmapped_district"] = unmapped(arrest_rows)
        arrests_block = {
            "window": ARRESTS_WINDOW,
            "table": skew_table(by_d(crimes_long), {"arrests": by_d(arrest_rows)},
                                district_meta),
            "group": group_ratio(by_d(crimes_long), by_d(arrest_rows), district_meta),
            "violent_group": group_ratio(by_d(violent_long), by_d(arrest_rows), district_meta),
        }

    return {
        "meta": {
            "definition": CRIME_BASELINE_DEFINITION,
            "victim_offenses": VICTIM_OFFENSES,
            "stops_window": STOPS_WINDOW,
            "stop_batch": stop_batch,
            "arrest_batch": arrest_batch,
        },
        "stops": {"window": STOPS_WINDOW, "table": table_now,
                  "group": group_ratio(by_d(crimes_now), numerators["stops"], district_meta),
                  "frisk_group": group_ratio(by_d(crimes_now), numerators["frisks"],
                                             district_meta),
                  # Robustness: the same group ratios against violent crime only.
                  "violent_group": group_ratio(by_d(violent_now), numerators["stops"],
                                               district_meta),
                  "violent_frisk_group": group_ratio(by_d(violent_now), numerators["frisks"],
                                                     district_meta),
                  "frisk_hits": frisk_hits},
        "arrests": arrests_block,
        "exclusions": exclusions,
    }
