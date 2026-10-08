#!/usr/bin/env python3
"""Load the district-population denominator into Postgres (manual, additive).

    python scripts/load_district_population.py

Persists the ALREADY-AUDITED output of the POC (scripts/build_district_population.py,
results in docs/district-population-findings.md) into two queryable tables:
  * ``district_population``  — 7 rows, one per MPD police district.
  * ``citywide_population``  — 1 row, the citywide baseline the representation
    index compares against.

It reuses the POC's aggregation verbatim (tract-centroid → district, via
database.district_population.aggregate) — it does NOT re-derive a new method. The
seven district rows plus the citywide total must reconcile to ~672,079 / 43.3%
Black, matching the findings doc.

Additive and manual by design: the DDL is created on first run (kept OUT of app
startup / ensure_schema). Nothing here touches crime_index.py, the report modules,
the frontend, or the existing tables. NOT wired into any cron.

Requires: DATABASE_URL. CENSUS_API_KEY is optional (keyless DCGIS fallback).
"""
from __future__ import annotations

import asyncio
import sys

from config import settings as _cfg
from database import db as _db
from database.district_population import (
    DC_TOTAL_EXPECTED,
    aggregate,
    create_district_population_schema,
    upsert_citywide_population,
    upsert_district_population,
)
from services.district_population_fetcher import fetch_districts, fetch_tracts


def _print_rows(rows: list[dict], citywide: dict) -> None:
    print(f"\n{'DIST':<5}{'POLICE':<8}{'tracts':>7}{'total pop':>12}{'Black pop':>12}{'% Black':>9}")
    for r in rows:
        print(f"{r['district']:<5}{str(r['police_district']):<8}{r['tracts']:>7}"
              f"{r['total_population']:>12,}{r['black_population']:>12,}{r['pct_black']:>8.1f}%")
    print(f"{'ALL':<5}{'':<8}{sum(r['tracts'] for r in rows):>7}"
          f"{citywide['total_population']:>12,}{citywide['black_population']:>12,}"
          f"{citywide['pct_black']:>8.1f}%")


def _reconcile(rows: list[dict], citywide: dict, orphans: list[dict]) -> bool:
    """Loud reconciliation, matching the findings doc. Returns True if it holds."""
    print("\n── Reconciliation ──")
    district_sum = sum(r["total_population"] for r in rows)
    dc_total = citywide["total_population"]
    orphan_pop = sum(t["total"] for t in orphans)
    disc = dc_total - district_sum
    lo, hi = DC_TOTAL_EXPECTED
    in_band = lo <= dc_total <= hi
    print(f"Citywide total (all tracts)      : {dc_total:>10,}")
    print(f"Summed across the 7 districts     : {district_sum:>10,}")
    print(f"Discrepancy (citywide - district) : {disc:>+10,}  "
          f"({100 * disc / dc_total:+.2f}%)" if dc_total else "  (n/a)")
    print(f"Citywide %Black (the baseline)    : {citywide['pct_black']:.1f}%")
    print(f"Orphan tracts (no district)       : {len(orphans)} ({orphan_pop:,} people)")
    print(f"Sanity: DC total in {lo:,}-{hi:,}? {'YES' if in_band else 'NO'}")
    holds = (disc == orphan_pop) and in_band and len(rows) == 7
    print(f"RECONCILIATION HOLDS: {'YES' if holds else 'NO'}")
    return holds


async def main() -> int:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set.", file=sys.stderr)
        return 1

    print("Fetching MPD police-district boundaries + ACS tract population…", flush=True)
    districts = fetch_districts()
    print(f"  districts: {len(districts)}", flush=True)
    tracts, source_label = fetch_tracts()
    print(f"  tracts: {len(tracts)} | source: {source_label}", flush=True)
    if len(districts) != 7:
        print(f"ERROR: expected 7 districts, got {len(districts)}.", file=sys.stderr)
        return 1

    rows, citywide, orphans = aggregate(tracts, districts)
    _print_rows(rows, citywide)
    holds = _reconcile(rows, citywide, orphans)

    pool = await _db.get_pool()
    async with pool.acquire() as conn:
        await create_district_population_schema(conn)
        n = await upsert_district_population(conn, rows, source_label=source_label)
        await upsert_citywide_population(conn, citywide, source_label=source_label)
        print(f"\nUpserted {n} district rows + 1 citywide baseline row.", flush=True)

        # Read back what actually landed (the persisted denominator).
        print("\n── Persisted district_population ──")
        for r in await conn.fetch(
            "SELECT district, police_district, total_population, black_population, pct_black "
            "FROM district_population ORDER BY district"):
            print(f"  district {r['district']} ({r['police_district']}): "
                  f"{r['total_population']:,} pop, {r['black_population']:,} Black, "
                  f"{float(r['pct_black']):.1f}%")
        cw = await conn.fetchrow(
            "SELECT total_population, black_population, pct_black, acs_vintage "
            "FROM citywide_population WHERE scope='dc'")
        print(f"  citywide (dc): {cw['total_population']:,} pop, {cw['black_population']:,} "
              f"Black, {float(cw['pct_black']):.1f}% | ACS {cw['acs_vintage']}")

    await _db.close_pool()
    print("\nDistrict-population load complete.", flush=True)
    return 0 if holds else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
