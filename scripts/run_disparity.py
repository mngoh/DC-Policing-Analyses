#!/usr/bin/env python3
"""Run the disparity engine against the database and print readable tables.

    python scripts/run_disparity.py

Read-only runner (same pattern as scripts/load_district_population.py and
scripts/ingest_stops.py): connects via DATABASE_URL, calls the functions in
database/disparity.py, and prints the citywide summary, the per-district
representation index (LOCAL baseline primary + CITYWIDE baseline secondary, all 7
side by side), the exclusions summary, and the preliminary outcome split. It does
NOT modify the engine, add routes, create schema, or write anything - SELECTs only.

Requires DATABASE_URL, and that the numerator + denominator are already loaded:
  * stops           - scripts/ingest_stops.py
  * district_population + citywide_population - scripts/load_district_population.py
If either is missing the engine raises a clear error, surfaced below.
"""
from __future__ import annotations

import asyncio
import sys

import asyncpg

from config import settings as _cfg
from database import db as _db
from database.disparity import (
    citywide_disparity_summary,
    district_representation_index,
    stop_outcome_split,
)


def _pct(x, digits=1):
    return f"{x * 100:.{digits}f}%" if x is not None else "   n/a"


def _idx(x):
    return f"{x:.2f}x" if x is not None else "  n/a"


def _num(x):
    return f"{x:,}" if x is not None else "n/a"


def _print_citywide(summary: dict) -> None:
    m, c = summary["meta"], summary["citywide"]
    print("\n" + "=" * 82)
    print(f"CITYWIDE REPRESENTATION SUMMARY   (batch {m['ingest_batch']}, layer {m['source_layer']})")
    print("=" * 82)
    print(f"  Black stop share (known race) : {_pct(c['black_stop_share'])}")
    print(f"  Citywide Black population base : {_pct(c['citywide_black_pop_share'])}")
    print(f"  Representation index          : {_idx(c['representation_index'])}")
    print(f"  Black stops / known-race stops: {_num(c['black_stops'])} / {_num(c['known_race_stops'])}")
    print(f"  Grain                         : {c['grain']}")


def _print_districts(index: dict) -> None:
    print("\n" + "=" * 82)
    print("PER-DISTRICT REPRESENTATION INDEX   (LOCAL baseline = primary; CITYWIDE = comparison)")
    print("=" * 82)
    hdr = (f"{'DIST':<5}{'POLICE':<7}{'BLACK SHARE':>12}{'LOCAL POP%':>11}"
           f"{'LOCAL idx':>11}{'CITYWIDE idx':>13}{'BLACK STOPS(N)':>16}{'KNOWN-RACE':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in index["districts"]:
        print(f"{r['district']:<5}{str(r['police_district']):<7}"
              f"{_pct(r['black_stop_share']):>12}{_pct(r['local_black_pop_share']):>11}"
              f"{_idx(r['local_representation_index']):>11}{_idx(r['representation_index']):>13}"
              f"{_num(r['black_stops']):>16}{_num(r['known_race_stops']):>12}")
    print("\n  LOCAL idx    = Black stop share / this district's OWN Black population share (primary)")
    print("  CITYWIDE idx = Black stop share / citywide Black population share (~43.3%, comparison)")
    print("  BLACK STOPS(N) is the numerator the LOCAL ratio rests on - caveat thin bases (low-Black districts).")


def _print_exclusions(index: dict) -> None:
    e = index["exclusions"]
    print("\n" + "=" * 82)
    print("EXCLUSIONS (loud - never silently dropped)")
    print("=" * 82)
    print(f"  Unknown race       : {_num(e['unknown_race_stops']):>10}  ({e['unknown_race_pct']}%)"
          "  - excluded from known-race denominator")
    print(f"  Unmapped district  : {_num(e['unmapped_district_stops']):>10}  ({e['unmapped_district_pct']}%)"
          "  - excluded from per-district metrics")
    print(f"  Total stops        : {_num(e['total_stops']):>10}")
    print(f"  Reported coverage  : {e['reported_coverage_pct']}% of all stops (mapped + known race)")


def _print_outcome(outcome: dict) -> None:
    print("\n" + "=" * 82)
    print("PRELIMINARY OUTCOME SPLIT - no-enforcement share (Harbor excluded)  [PRELIMINARY]")
    print("=" * 82)
    hdr = (f"{'DIST':<5}{'POLICE':<8}{'BLACK no-enf':>14}{'NON-BLK no-enf':>16}"
           f"{'BLACK street':>14}{'NON-BLK street':>16}{'HARBOR excl':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in outcome["districts"]:
        b, nb = r["black"], r["non_black_known_race"]
        harbor = b["harbor_excluded"] + nb["harbor_excluded"]
        print(f"{r['district']:<5}{str(r['police_district']):<8}"
              f"{_pct(b['no_enforcement_share']):>14}{_pct(nb['no_enforcement_share']):>16}"
              f"{_num(b['street_total']):>14}{_num(nb['street_total']):>16}{_num(harbor):>13}")
    eh = outcome["exclusions"]
    print(f"\n  Harbor stops excluded from split: {_num(eh['harbor_stops_excluded'])} "
          f"({eh['harbor_stops_excluded_pct']}% of all stops)")
    print(f"  {outcome['meta']['definition']}")


async def main() -> int:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set - cannot run the disparity engine.",
              file=sys.stderr)
        print("Run against your database with:\n"
              "  DATABASE_URL='<your-postgres-url>' python scripts/run_disparity.py",
              file=sys.stderr)
        return 1

    pool = await _db.get_pool()
    try:
        async with pool.acquire() as conn:
            summary = await citywide_disparity_summary(conn)
            index = await district_representation_index(conn)
            outcome = await stop_outcome_split(conn)
    except RuntimeError as exc:
        # Engine's own guard (no finished batch / baseline not loaded).
        print(f"ERROR: {exc}", file=sys.stderr)
        await _db.close_pool()
        return 2
    except asyncpg.exceptions.UndefinedTableError:
        # The numerator/denominator tables don't exist in this database yet.
        print("ERROR: the disparity source tables are not present in this database.\n"
              "Load them first (both are additive, manual, read-only to run):\n"
              "  1. python scripts/ingest_stops.py            # numerator: stops\n"
              "  2. python scripts/load_district_population.py # denominator: district+citywide population\n"
              "then re-run this script.", file=sys.stderr)
        await _db.close_pool()
        return 2

    _print_citywide(summary)
    _print_districts(index)
    _print_exclusions(index)
    _print_outcome(outcome)
    print()

    await _db.close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
