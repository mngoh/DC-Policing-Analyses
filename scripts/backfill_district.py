#!/usr/bin/env python3
"""Backfill missing crimes.district via point-in-polygon (one-time data fix).

    DATABASE_URL=... python scripts/backfill_district.py

About 1,500 archive rows (598k total) have coordinates but no district; 336
of them are a single November 2024 ingest batch. The district theft-exit
analysis (database/theft_exit.py) estimates monthly district series through
that month, so the gap is closed BEFORE estimation and the recovered-row
count is reported on the page.

Polygons come from the same DCGIS Police Districts layer used by
scripts/build_district_polygons.py; matching is PostGIS ST_Covers on the
existing crimes.location POINT column (SRID 4326). Rows whose points fall
outside every district polygon (bad geocodes, out-of-city points) are left
NULL and counted in the summary.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request

import asyncpg

DISTRICTS_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/9"
)


def fetch_district_geojson() -> list[tuple[str, str]]:
    """(district, geojson geometry) for the 7 police districts."""
    q = {"where": "1=1", "outFields": "DISTRICT", "returnGeometry": "true",
         "outSR": "4326", "f": "geojson"}
    url = DISTRICTS_LAYER + "/query?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=90) as r:  # noqa: S310 fixed DCGIS host
        data = json.load(r)
    out = []
    for f in data["features"]:
        dist = str(int(f["properties"]["DISTRICT"]))
        out.append((dist, json.dumps(f["geometry"])))
    if len(out) != 7:
        raise RuntimeError(f"expected 7 district polygons, got {len(out)}")
    return out


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL env var not set", file=sys.stderr)
        raise SystemExit(2)
    polys = fetch_district_geojson()
    conn = await asyncpg.connect(url, timeout=30)
    try:
        before = await conn.fetchval(
            "SELECT count(*) FROM crimes WHERE district IS NULL")
        await conn.execute("""
            CREATE TEMP TABLE district_poly (
                district text PRIMARY KEY,
                geom geometry(MultiPolygon, 4326)
            )""")
        for dist, gj in polys:
            await conn.execute("""
                INSERT INTO district_poly (district, geom)
                VALUES ($1, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($2), 4326)))
            """, dist, gj)
        updated = await conn.execute("""
            UPDATE crimes c
            SET district = p.district
            FROM district_poly p
            WHERE c.district IS NULL
              AND c.location IS NOT NULL
              AND ST_Covers(p.geom, c.location)
        """)
        recovered = int(updated.split()[-1])
        after = await conn.fetchval(
            "SELECT count(*) FROM crimes WHERE district IS NULL")
        by_month = await conn.fetch("""
            SELECT date_trunc('month', report_dat AT TIME ZONE 'UTC')::date m,
                   count(*) n
            FROM crimes
            WHERE district IS NULL
            GROUP BY 1 ORDER BY n DESC LIMIT 5
        """)
        print(json.dumps({
            "null_district_before": before,
            "recovered_rows": recovered,
            "null_district_after": after,
            "largest_remaining_gaps": [
                {"month": str(r["m"]), "n": r["n"]} for r in by_month],
        }, indent=2))
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
