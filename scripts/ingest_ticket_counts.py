#!/usr/bin/env python3
"""Manual ingest of camera-vs-officer moving-violation counts (2023-2025).

    python scripts/ingest_ticket_counts.py

Per month (36 months): one server-side GROUP BY LOCATION for camera tickets
(fixed sites, ~350/month, avg lat/lng per site) + raw lat/lng rows for MPD
officer tickets (~3-4k/month) + one agency-code GROUP BY for the excluded
'other agency' count. District assignment is client-side point-in-polygon
against the real MPD district polygons. Stores (month, source, district)
counts as one batch snapshot. ASCII-only output. Requires DATABASE_URL.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date

import httpx

from config import settings as _cfg
from database import db as _db
from database.tickets import (
    CAMERA_AGENCY_CODE, DISTRICT_POLYGONS_URL, OFFICER_AGENCY_CODES,
    assign_district, create_tickets_schema, finish_batch, month_layer_url,
    start_batch, upsert_counts,
)
from services.stops_fetcher import fetch_stops as fetch_rows

YEARS = (2023, 2024, 2025)
_COUNT_STAT = json.dumps([
    {"statisticType": "count", "onStatisticField": "OBJECTID", "outStatisticFieldName": "n"},
    {"statisticType": "avg", "onStatisticField": "LATITUDE", "outStatisticFieldName": "lat"},
    {"statisticType": "avg", "onStatisticField": "LONGITUDE", "outStatisticFieldName": "lng"},
])
_AGENCY_STAT = json.dumps([
    {"statisticType": "count", "onStatisticField": "OBJECTID", "outStatisticFieldName": "n"},
])


async def _get_json(client, url: str, params: dict) -> dict:
    """GET with the same 3-attempt retry policy as the row fetcher; DC's ArcGIS
    times out transiently and a 30-minute sweep must survive that."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = await client.get(url, params=params)
            if r.status_code < 500:
                return r.json()
        except httpx.TransportError as exc:
            last = exc
        await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"ArcGIS unavailable after retries: {url} ({last})")


async def _fetch_polygons(client) -> dict[int, list]:
    body = await _get_json(client, DISTRICT_POLYGONS_URL + "/query", {
        "where": "1=1", "outFields": "DISTRICT", "returnGeometry": "true",
        "outSR": "4326", "f": "json"})
    polys = {f["attributes"]["DISTRICT"]: f["geometry"]["rings"]
             for f in body["features"]}
    assert sorted(polys) == [1, 2, 3, 4, 5, 6, 7], f"expected 7 districts, got {sorted(polys)}"
    return polys


async def _month_counts(client, year: int, month: int, polygons) -> list[tuple]:
    """(month, source, district, n) tuples for one month."""
    url = month_layer_url(year, month) + "/query"
    first = date(year, month, 1)

    # Camera: fixed sites, grouped server-side.
    body = await _get_json(client, url, {
        "where": f"ISSUING_AGENCY_CODE={CAMERA_AGENCY_CODE}",
        "groupByFieldsForStatistics": "LOCATION",
        "outStatistics": _COUNT_STAT, "f": "json"})
    if "error" in body:
        raise RuntimeError(f"ArcGIS error {year}-{month:02d} camera: {body['error']}")
    cam: dict = {}
    cam_unassigned = 0
    for f in body.get("features", []):
        # ArcGIS uppercases outStatisticFieldName on some layers -> normalize.
        a = {k.lower(): v for k, v in f["attributes"].items()}
        d = assign_district(a.get("lng"), a.get("lat"), polygons)
        if d is None:
            cam_unassigned += a["n"]
        else:
            cam[d] = cam.get(d, 0) + a["n"]

    # Officer: raw rows (low volume), assigned per ticket.
    codes = ",".join(str(c) for c in OFFICER_AGENCY_CODES)
    officer_rows = await fetch_rows(
        layer_url=month_layer_url(year, month),
        where=f"ISSUING_AGENCY_CODE IN ({codes})",
        fields="OBJECTID,LATITUDE,LONGITUDE")
    off: dict = {}
    off_unassigned = 0
    for a in officer_rows:
        d = assign_district(a.get("LONGITUDE"), a.get("LATITUDE"), polygons)
        if d is None:
            off_unassigned += 1
        else:
            off[d] = off.get(d, 0) + 1

    # Other agencies: excluded, counted.
    body = await _get_json(client, url, {
        "where": f"ISSUING_AGENCY_CODE NOT IN ({codes},{CAMERA_AGENCY_CODE}) "
                 "OR ISSUING_AGENCY_CODE IS NULL",
        "outStatistics": _AGENCY_STAT, "f": "json"})
    other_attrs = {k.lower(): v for k, v in
                   ((body.get("features") or [{}])[0].get("attributes") or {}).items()}
    other = other_attrs.get("n", 0) or 0

    rows: list[tuple] = []
    for d, n in cam.items():
        rows.append((first, "camera", d, n))
    for d, n in off.items():
        rows.append((first, "officer", d, n))
    if cam_unassigned:
        rows.append((first, "camera", None, cam_unassigned))
    if off_unassigned:
        rows.append((first, "officer", None, off_unassigned))
    if other:
        rows.append((first, "other_agency", None, other))
    print(f"  {year}-{month:02d}: camera {sum(cam.values()):,} (+{cam_unassigned} unassigned)  "
          f"officer {sum(off.values()):,} (+{off_unassigned})  other {other:,}", flush=True)
    return rows


async def main() -> int:
    if not _cfg.DATABASE_URL:
        print("ERROR: DATABASE_URL env var not set.", file=sys.stderr)
        return 1

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        polygons = await _fetch_polygons(client)
        print(f"Fetched {len(polygons)} district polygons.", flush=True)
        all_rows: list[tuple] = []
        months = 0
        for year in YEARS:
            for month in range(1, 13):
                all_rows.extend(await _month_counts(client, year, month, polygons))
                months += 1
    print(f"Collected {len(all_rows)} count rows for {months} months "
          f"in {time.monotonic() - t0:.0f}s", flush=True)

    pool = await _db.get_pool()
    async with pool.acquire() as conn:
        await create_tickets_schema(conn)
        batch_id = await start_batch(conn, note=f"manual ingest_ticket_counts.py, {YEARS}")
        rows = [(batch_id, *r) for r in all_rows]
        n = await upsert_counts(conn, rows)
        await finish_batch(conn, batch_id, months=months)
        print(f"Upserted {n} rows as batch {batch_id}.", flush=True)

        print("\n-- Pooled audit --", flush=True)
        for r in await conn.fetch(
            "SELECT source, district, SUM(n) AS n FROM moving_violation_counts "
            "WHERE ingest_batch=$1 GROUP BY source, district "
            "ORDER BY source, district NULLS LAST", batch_id):
            print(f"  {r['source']:13} district {str(r['district']):>4}: {r['n']:,}")

    await _db.close_pool()
    print("\nTicket-count ingestion complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
