#!/usr/bin/env python3
"""Day/night split of the district local representation index (analysis runner).

    python scripts/run_disparity_timesplit.py

Additive, read-only exploration: does splitting stops by time of day reveal
anything before we decide to surface it? Prints, for the current snapshot:
  1. A DATA-QUALITY check first: what share of stop rows have no usable time
     (NULL/unparseable stop_datetime) and so fall out of BOTH day and night.
  2. Three per-district local-index tables side by side — ALL-HOURS (baseline),
     DAY, and NIGHT — each showing the local index, Black stop share, and the
     Black stop N the cell rests on.
  3. Thin cells (Black stop N below the threshold below) marked low-confidence,
     because splitting by time cuts the sample.

night = 22:00-04:59, day = everything else, hour taken in DC LOCAL time
(America/New_York) since stop_datetime is stored UTC. Does not modify the
/districts pages or the choropleth.

Requires DATABASE_URL.
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
    time_coverage,
)

# Below this many Black stops, a district-window cell is flagged low-confidence.
THIN_N = 300

WINDOWS = [(None, "ALL-HOURS"), ("day", "DAY 05-22"), ("night", "NIGHT 22-05")]


def _idx(x):
    return f"{x:.2f}x" if x is not None else "  n/a"


def _pct(x):
    return f"{x * 100:.1f}%" if x is not None else " n/a"


def _num(x):
    return f"{x:,}" if x is not None else "n/a"


def _cell(row: dict) -> tuple[str, bool]:
    """Format one window cell 'idx  share      N' and whether it is thin."""
    n = row.get("black_stops") if row else None
    thin = n is not None and n < THIN_N
    flag = "!" if thin else " "
    idx = _idx(row.get("local_representation_index")) if row else "  n/a"
    share = _pct(row.get("black_stop_share")) if row else " n/a"
    return f"{idx:>7} {share:>6} {_num(n):>8}{flag}", thin


def _print_quality(cov: dict) -> None:
    print("\n" + "=" * 78)
    print(f"DATA QUALITY  (batch {cov['ingest_batch']}, timezone {cov['timezone']} = DC local)")
    print("=" * 78)
    print(f"  Total stops                    : {_num(cov['total_stops'])}")
    print(f"  NULL / unparseable stop_datetime: {_num(cov['null_time_stops'])} "
          f"({cov['null_time_pct']}%)  -> excluded from BOTH day and night")
    print(f"  Timed rows usable for the split : day={_num(cov['day_stops'])}, "
          f"night={_num(cov['night_stops'])}")


def _print_table(by_window: dict, cw_by_window: dict) -> list:
    print("\n" + "=" * 78)
    print("PER-DISTRICT LOCAL INDEX BY TIME WINDOW   (night = 22:00-04:59 DC local)")
    print(f"cell = local-index  Black-share  Black-N ;  '!' = Black N < {THIN_N} (low confidence)")
    print("=" * 78)
    head = f"{'DIST':<5}"
    sub = f"{'':<5}"
    for _w, label in WINDOWS:
        head += f"|{label:^24}"
        sub += f"|{'idx':>7} {'share':>6} {'N':>8} "
    print(head)
    print(sub)
    print("-" * len(sub))

    thin_cells = []
    districts = sorted({d["district"] for d in by_window[None]["districts"]})
    for dnum in districts:
        line = f"{dnum:<5}"
        for w, _label in WINDOWS:
            row = next((r for r in by_window[w]["districts"] if r["district"] == dnum), {})
            text, thin = _cell(row)
            line += "|" + text
            if thin:
                thin_cells.append((dnum, _label_for(w)))
        print(line)

    # Citywide anchor row (all districts incl unmapped).
    cw_line = f"{'DC':<5}"
    for w, _label in WINDOWS:
        c = cw_by_window[w]["citywide"]
        cw_line += "|" + f"{_idx(c.get('representation_index')):>7} {_pct(c.get('black_stop_share')):>6} {_num(c.get('black_stops')):>8} "
    print("-" * len(sub))
    print(cw_line + "   (citywide anchor)")
    return thin_cells


def _label_for(w) -> str:
    return {None: "all", "day": "day", "night": "night"}[w]


def _print_read(by_window: dict, thin_cells: list) -> None:
    print("\n" + "=" * 78)
    print("READ")
    print("=" * 78)
    d2 = {w: next((r for r in by_window[w]["districts"] if r["district"] == 2), {}) for w, _ in WINDOWS}
    a = d2[None].get("local_representation_index")
    ni = d2["night"].get("local_representation_index")
    nn = d2["night"].get("black_stops")
    if a and ni:
        direction = "HIGHER" if ni > a * 1.05 else "LOWER" if ni < a * 0.95 else "SIMILAR"
        print(f"  District 2 local index: all-hours {_idx(a)} -> night {_idx(ni)}  ({direction}); "
              f"night rests on {_num(nn)} Black stops{'  [THIN]' if nn and nn < THIN_N else ''}")
    if thin_cells:
        joined = ", ".join(f"D{d}/{w}" for d, w in thin_cells)
        print(f"  Thin cells (Black N < {THIN_N}, treat as noise): {joined}")
    else:
        print(f"  No district-window cell falls below {THIN_N} Black stops.")


async def main() -> int:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set - cannot run the time split.", file=sys.stderr)
        print("Run against your database with:\n"
              "  DATABASE_URL='<your-postgres-url>' python scripts/run_disparity_timesplit.py",
              file=sys.stderr)
        return 1

    pool = await _db.get_pool()
    try:
        async with pool.acquire() as conn:
            cov = await time_coverage(conn)
            by_window = {w: await district_representation_index(conn, time_window=w) for w, _ in WINDOWS}
            cw_by_window = {w: await citywide_disparity_summary(conn, time_window=w) for w, _ in WINDOWS}
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        await _db.close_pool()
        return 2
    except asyncpg.exceptions.UndefinedTableError:
        print("ERROR: disparity source tables not present. Load them first:\n"
              "  python scripts/ingest_stops.py\n"
              "  python scripts/load_district_population.py", file=sys.stderr)
        await _db.close_pool()
        return 2

    _print_quality(cov)
    thin_cells = _print_table(by_window, cw_by_window)
    _print_read(by_window, thin_cells)
    print()
    await _db.close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
