#!/usr/bin/env python3
"""Manual ingest of MPD Adult Arrests (layer 38) into the ``arrests`` table.

    python scripts/ingest_arrests.py

Same contract as scripts/ingest_stops.py: additive, manual, snapshot-per-run
batches; "current" arrests are the latest FINISHED batch. ASCII-only output.
Requires DATABASE_URL. Touches only the two arrest tables.
"""
from __future__ import annotations

import asyncio
import sys
import time

from config import settings as _cfg
from database import db as _db
from database.arrests import (
    ARREST_FIELDS, ARRESTS_LAYER_URL,
    create_arrests_schema, start_batch, finish_batch, transform_arrest, upsert_arrests,
)
from services.stops_fetcher import fetch_stops as fetch_features

_UPSERT_BATCH = 5000


async def _report(conn, batch_id: int) -> None:
    print(f"\n-- Ingested-row audit (batch {batch_id}) --", flush=True)
    total = await conn.fetchval("SELECT COUNT(*) FROM arrests WHERE ingest_batch=$1", batch_id)
    print(f"Rows in this batch: {total:,}")

    print("\nRows per year:")
    for r in await conn.fetch(
        "SELECT year, COUNT(*) AS n FROM arrests WHERE ingest_batch=$1 GROUP BY year ORDER BY year", batch_id):
        print(f"  {r['year']}: {r['n']:,}")

    print("\nRows per normalized district (NULL = unmapped):")
    for r in await conn.fetch(
        "SELECT district, COUNT(*) AS n FROM arrests WHERE ingest_batch=$1 "
        "GROUP BY district ORDER BY district NULLS LAST", batch_id):
        print(f"  district {str(r['district']):>4}: {r['n']:,}")

    print("\nRace (verbatim; UNK and UNKNOWN are both unknown labels):")
    for r in await conn.fetch(
        "SELECT race, COUNT(*) AS n FROM arrests WHERE ingest_batch=$1 "
        "GROUP BY race ORDER BY n DESC", batch_id):
        print(f"  {str(r['race']):16} {r['n']:,}")


async def main() -> int:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set.", file=sys.stderr)
        return 1

    print(f"Fetching MPD Adult Arrests (layer 38)\n  {ARRESTS_LAYER_URL}", flush=True)
    t0 = time.monotonic()
    attrs = await fetch_features(
        layer_url=ARRESTS_LAYER_URL, fields=ARREST_FIELDS,
        on_page=lambda p, f, t: print(f"  page {p}: +{f} (total {t:,})", flush=True))
    print(f"Fetched {len(attrs):,} records in {time.monotonic() - t0:.1f}s", flush=True)

    pool = await _db.get_pool()
    async with pool.acquire() as conn:
        await create_arrests_schema(conn)
        batch_id = await start_batch(conn, note="manual ingest_arrests.py, layer 38")
        print(f"Opened batch {batch_id} (snapshot).", flush=True)

        rows = [t for a in attrs if (t := transform_arrest(a, ingest_batch=batch_id)) is not None]
        skipped = len(attrs) - len(rows)
        if skipped:
            print(f"  ({skipped} records skipped - missing GLOBALID)", flush=True)

        print(f"Upserting {len(rows):,} rows for batch {batch_id}...", flush=True)
        t1 = time.monotonic()
        processed = 0
        for i in range(0, len(rows), _UPSERT_BATCH):
            processed += await upsert_arrests(conn, rows[i:i + _UPSERT_BATCH])
        distinct_gids = len({r[1] for r in rows})
        await finish_batch(conn, batch_id, fetched_rows=len(attrs),
                           upserted_rows=processed, distinct_globalids=distinct_gids)
        print(f"Upserted {processed:,} rows in {time.monotonic() - t1:.1f}s", flush=True)
        await _report(conn, batch_id)

    await _db.close_pool()
    print("\nArrests ingestion complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
