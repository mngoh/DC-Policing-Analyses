#!/usr/bin/env python3
"""Manual ingest of MPD Stop Data into the ``stops`` table.

    python scripts/ingest_stops.py               # layer 41 (2023-present), the default
    python scripts/ingest_stops.py --layer 45    # layer 45 (2019-07-22 to 2022)

Additive and manual by design. Each run records a new BATCH (snapshot) and tags
every row with it — so this is NOT "idempotent" in the no-new-rows sense: a fresh
run re-lands the current MPD release as a new snapshot. "Current" stops are the
LATEST batch; re-running the same snapshot is idempotent within its batch. This
is deliberate — Stop Data refreshes only ~SEMI-ANNUALLY and MPD may regenerate
GLOBALIDs on republish, so snapshot-per-release survives that (see the module
docstring and docs/stops-ingestion-findings.md). NOT wired into any cron;
scheduling is a later, separate decision. The DDL is created on first run
(kept out of app startup / ensure_schema).

Requires: DATABASE_URL. Does not touch crime_index.py, the report modules, the
frontend, or the existing crimes table.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from config import settings as _cfg
from database import db as _db
from database.stops import (
    CURRENT_LAYER, LAYER_FIELDS, LAYER_URLS,
    create_stops_schema, start_batch, finish_batch, transform_stop, upsert_stops,
)
from services.stops_fetcher import fetch_stops

_UPSERT_BATCH = 5000


async def _report(conn, batch_id: int, source_layer: int) -> None:
    # ASCII-only output: box-drawing characters crash cp1252 Windows consoles
    # (the ingest itself finishes first, but the audit then dies mid-print).
    print(f"\n-- Ingested-row audit (batch {batch_id}) --", flush=True)
    total = await conn.fetchval("SELECT COUNT(*) FROM stops WHERE ingest_batch=$1", batch_id)
    print(f"Rows in this batch: {total:,}")

    print("\nRows per year:")
    for r in await conn.fetch(
        "SELECT year, COUNT(*) AS n FROM stops WHERE ingest_batch=$1 GROUP BY year ORDER BY year", batch_id):
        print(f"  {r['year']}: {r['n']:,}")

    print("\nRows per normalized district (NULL = unmapped):")
    for r in await conn.fetch(
        "SELECT district, COUNT(*) AS n FROM stops WHERE ingest_batch=$1 "
        "GROUP BY district ORDER BY district NULLS LAST", batch_id):
        print(f"  district {str(r['district']):>4}: {r['n']:,}")
    unmapped = await conn.fetchval(
        "SELECT COUNT(*) FROM stops WHERE ingest_batch=$1 AND district_unmapped", batch_id)
    print(f"  flagged district_unmapped: {unmapped:,}")

    print("\nRace ('Unknown' preserved as a real category):")
    for r in await conn.fetch(
        "SELECT ethnicity, COUNT(*) AS n FROM stops WHERE ingest_batch=$1 "
        "GROUP BY ethnicity ORDER BY n DESC", batch_id):
        print(f"  {str(r['ethnicity']):32} {r['n']:,}")
    sql_null = await conn.fetchval(
        "SELECT COUNT(*) FROM stops WHERE ingest_batch=$1 AND ethnicity IS NULL", batch_id)
    print(f"  (SQL NULL race): {sql_null:,}")

    # Cross-batch drift: GLOBALID overlap vs the previous batch for this layer.
    prev = await conn.fetchval(
        "SELECT MAX(batch_id) FROM stop_ingest_batches "
        "WHERE source_layer=$1 AND batch_id<$2 AND finished_at IS NOT NULL",
        source_layer, batch_id)
    if prev is not None:
        overlap = await conn.fetchval(
            "SELECT COUNT(*) FROM stops a JOIN stops b USING (global_id) "
            "WHERE a.ingest_batch=$1 AND b.ingest_batch=$2", batch_id, prev)
        prev_n = await conn.fetchval("SELECT COUNT(*) FROM stops WHERE ingest_batch=$1", prev)
        pct = (100 * overlap / prev_n) if prev_n else 0.0
        print(f"\nGLOBALID overlap with previous batch {prev}: {overlap:,}/{prev_n:,} = {pct:.1f}%")
        print("  (low overlap => MPD regenerated GLOBALIDs on republish; treat as a new snapshot)")

    print("\nBatch history:")
    for r in await conn.fetch(
        "SELECT batch_id, source_layer, upserted_rows, distinct_globalids, finished_at "
        "FROM stop_ingest_batches ORDER BY batch_id"):
        print(f"  batch {r['batch_id']} layer {r['source_layer']}: "
              f"{r['upserted_rows'] or 0:,} rows, {r['distinct_globalids'] or 0:,} distinct GIDs")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one MPD Stop Data layer as a new batch.")
    parser.add_argument("--layer", type=int, choices=sorted(LAYER_URLS), default=CURRENT_LAYER,
                        help="ArcGIS source layer: 41 = 2023-present (default), 45 = 2019-2022")
    layer = parser.parse_args().layer

    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set.", file=sys.stderr)
        return 1

    layer_url = LAYER_URLS[layer]
    print(f"Fetching MPD Stop Data (layer {layer})\n  {layer_url}", flush=True)
    t0 = time.monotonic()
    attrs = await fetch_stops(layer_url=layer_url, fields=LAYER_FIELDS[layer],
                              on_page=lambda p, f, t: print(f"  page {p}: +{f} (total {t:,})", flush=True))
    print(f"Fetched {len(attrs):,} records in {time.monotonic() - t0:.1f}s", flush=True)

    pool = await _db.get_pool()
    async with pool.acquire() as conn:
        await create_stops_schema(conn)
        batch_id = await start_batch(conn, source_layer=layer,
                                     note=f"manual ingest_stops.py, layer {layer}")
        print(f"Opened batch {batch_id} (snapshot).", flush=True)

        rows = [t for a in attrs
                if (t := transform_stop(a, source_layer=layer, ingest_batch=batch_id)) is not None]
        skipped = len(attrs) - len(rows)
        if skipped:
            print(f"  ({skipped} records skipped — missing GLOBALID)", flush=True)

        print(f"Upserting {len(rows):,} rows for batch {batch_id}…", flush=True)
        t1 = time.monotonic()
        processed = 0
        for i in range(0, len(rows), _UPSERT_BATCH):
            processed += await upsert_stops(conn, rows[i:i + _UPSERT_BATCH])
        distinct_gids = len({r[1] for r in rows})  # STOP_COLUMNS[1] == global_id
        await finish_batch(conn, batch_id, fetched_rows=len(attrs),
                           upserted_rows=processed, distinct_globalids=distinct_gids)
        print(f"Upserted {processed:,} rows in {time.monotonic() - t1:.1f}s", flush=True)
        await _report(conn, batch_id, layer)

    await _db.close_pool()
    print("\nStops ingestion complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
