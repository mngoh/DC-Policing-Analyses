"""
ingest.py — Bulk ingestion of DC MPD crime data (2008–2025) into PostgreSQL.

Usage:
    python database/ingest.py                  # ingest all years 2008–2025
    python database/ingest.py 2022 2023        # ingest specific years
    python database/ingest.py --year 2024      # ingest single year
    python database/ingest.py --allow-bulk-rewrite 2024
        # deliberate backfill: suppress the anomaly brake so a sync that
        # rewrites many existing rows is applied instead of blocked. The
        # NULL/empty and identity (ccn, city) guards still apply.

Safe to re-run — duplicates on (ccn, city) are not re-inserted; existing rows
receive MPD corrections through the guarded upsert in db.upsert_records
(NULL/empty incoming values never overwrite stored data, and an anomaly brake
blocks unexpectedly large rewrites unless --allow-bulk-rewrite is passed).

Requires: DATABASE_URL env var pointing to a PostgreSQL instance.
"""
from __future__ import annotations

import asyncio
import sys
import time

import httpx

from config import settings as _cfg
from database.constants import YEAR_LAYERS
from database.db import ensure_schema, upsert_records

FIELDS     = "CCN,OFFENSE,METHOD,SHIFT,REPORT_DAT,WARD,DISTRICT,BLOCK,NEIGHBORHOOD_CLUSTER"
PAGE_SIZE  = 1000   # stay under ArcGIS MaxRecordCount of 1000


def year_where(year: int) -> str:
    """Full-calendar-year ArcGIS filter, end-exclusive.

    ArcGIS DATE literals are midnight timestamps, so an inclusive
    `<= DATE 'YYYY-12-31'` cuts the year off at 00:00 on Dec 31 and silently
    drops any Dec-31 record carrying a time of day.
    """
    return f"REPORT_DAT >= DATE '{year}-01-01' AND REPORT_DAT < DATE '{year + 1}-01-01'"


async def fetch_year(year: int) -> list[dict]:
    url     = YEAR_LAYERS[year]
    where   = year_where(year)
    records = []
    offset  = 0
    page    = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        while True:
            page += 1
            params = {
                "where":             where,
                "outFields":         FIELDS,
                "resultRecordCount": PAGE_SIZE,
                "resultOffset":      offset,
                "returnGeometry":    "true",
                "outSR":             "4326",
                "f":                 "json",
            }

            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()

            if "error" in body:
                raise RuntimeError(f"ArcGIS error: {body['error']}")

            features = body.get("features", [])

            for feat in features:
                attrs = feat.get("attributes", {})
                geom  = feat.get("geometry") or {}
                attrs["LONGITUDE"] = geom.get("x")
                attrs["LATITUDE"]  = geom.get("y")
                records.append(attrs)

            print(
                f"  [{year}] page {page} — fetched {len(features)} records "
                f"(total so far: {len(records):,})",
                flush=True,
            )

            # ArcGIS sets exceededTransferLimit=True when there are MORE pages
            # Only stop when it's False (or absent) OR the page returned nothing
            if not features or not body.get("exceededTransferLimit", False):
                break

            offset += PAGE_SIZE

    return records


async def ingest_year(year: int, allow_bulk_rewrite: bool = False) -> None:
    print(f"\n[{year}] Starting fetch…", flush=True)
    t0      = time.monotonic()
    records = await fetch_year(year)
    fetched = len(records)
    elapsed = time.monotonic() - t0
    print(f"[{year}] Fetched {fetched:,} records in {elapsed:.1f}s — upserting…", flush=True)

    t1       = time.monotonic()
    inserted = await upsert_records(records, city="dc", allow_bulk_rewrite=allow_bulk_rewrite)
    elapsed2 = time.monotonic() - t1
    print(f"[{year}] Done — {inserted:,} rows upserted (dupes skipped) in {elapsed2:.1f}s", flush=True)


async def main(years: list[int], allow_bulk_rewrite: bool = False) -> None:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set.", file=sys.stderr)
        sys.exit(1)

    print("Ensuring schema…", flush=True)
    await ensure_schema()
    print("Schema ready.\n", flush=True)

    for year in sorted(years, reverse=True):
        if year not in YEAR_LAYERS:
            print(f"[{year}] Skipped — not in layer map", flush=True)
            continue
        try:
            await ingest_year(year, allow_bulk_rewrite=allow_bulk_rewrite)
        except Exception as exc:
            print(f"[{year}] ERROR: {exc}", file=sys.stderr)

    print("\nIngestion complete.", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]

    # Explicit opt-in for a deliberate backfill: suppresses the anomaly brake
    # in upsert_records (never the NULL/empty or identity guards). Default OFF.
    allow_bulk_rewrite = "--allow-bulk-rewrite" in args
    if allow_bulk_rewrite:
        args = [a for a in args if a != "--allow-bulk-rewrite"]
        print("Bulk-rewrite bypass ENABLED — anomaly brake suppressed for this run.", flush=True)

    if "--year" in args:
        idx   = args.index("--year")
        years = [int(args[idx + 1])]
    elif args and all(a.isdigit() for a in args):
        years = [int(a) for a in args]
    else:
        years = list(YEAR_LAYERS.keys())  # all 2008–2025

    asyncio.run(main(years, allow_bulk_rewrite=allow_bulk_rewrite))
