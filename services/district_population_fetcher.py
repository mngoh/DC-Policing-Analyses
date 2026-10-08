"""Fetchers for the district-population denominator (MPD district polygons + ACS
tract population), ported from the audited POC scripts/build_district_population.py.

Network-only; kept out of database/district_population.py so the aggregation and
its reconciliation test stay importable without network. Used solely by
scripts/load_district_population.py — not wired into the app or any cron.

Two sources, exactly as the POC:
  * MPD Police District boundary polygons (layer 9) — the district geometry.
  * ACS tract population — Census API primary (needs CENSUS_API_KEY, the
    ingest_census.py way: DP05_0001E total, DP05_0038PE %Black); the keyless
    DCGIS ACS-by-tract layer (layer 38, DP05_0001E + DP05_0038E Black count)
    as the fallback used when no key is present.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from database.district_population import ACS_TRACT_LAYER, CENSUS_API, DISTRICTS_LAYER


def _get(url: str, tries: int = 5):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:  # noqa: S310 - fixed DCGIS/Census hosts
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - surface anything, retry transient drops
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _q(layer: str, params: dict, tries: int = 5):
    params = {**params, "f": "json"}
    return _get(layer + "/query?" + urllib.parse.urlencode(params), tries)


def fetch_districts() -> list[dict]:
    """Return ``[{district, police_district, name, rings}]`` (lng/lat rings)."""
    d = _q(DISTRICTS_LAYER, {
        "where": "1=1",
        "outFields": "DISTRICT,POLICEDISTRICT,NAME",
        "returnGeometry": "true",
        "outSR": "4326",
    })
    out = []
    for f in d["features"]:
        a = f["attributes"]
        out.append({
            "district": int(a["DISTRICT"]),
            "police_district": a.get("POLICEDISTRICT"),
            "name": a.get("NAME"),
            "rings": (f.get("geometry") or {}).get("rings", []),
        })
    out.sort(key=lambda r: r["district"])
    return out


def fetch_tracts() -> tuple[list[dict], str]:
    """Return ``([{geoid, total, black, lng, lat}], source_label)``.

    Census API when CENSUS_API_KEY is set; keyless DCGIS layer 38 otherwise.
    """
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        try:
            tracts = _acs_from_census_api(key)
            return tracts, ("Census API (api.census.gov, ACS 2019-2023 5-year, "
                            "DP05_0001E/DP05_0038PE) + DCGIS tract geometry")
        except Exception as e:  # noqa: BLE001
            print(f"Census API path failed ({e!r}); falling back to DCGIS.", flush=True)
    else:
        print("No CENSUS_API_KEY — using the keyless DCGIS fallback (layer 38).", flush=True)
    tracts = _acs_from_dcgis()
    return tracts, ("DCGIS 'ACS 5-Year Demographic Characteristics of DC Census "
                    "Tracts' (layer 38, DC Office of Planning, ACS 2019-2023 5-year, "
                    "DP05_0001E/DP05_0038E)")


def _acs_from_dcgis() -> list[dict]:
    d = _q(ACS_TRACT_LAYER, {
        "where": "1=1",
        "outFields": "GEOID,DP05_0001E,DP05_0038E",
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": 400,
    })
    out = []
    for f in d["features"]:
        a = f["attributes"]
        rings = (f.get("geometry") or {}).get("rings", [])
        if not rings:
            continue
        total = int(a.get("DP05_0001E") or 0)
        black = int(a.get("DP05_0038E") or 0)  # DP05_0038E = Black-alone count
        lng, lat = _centroid(rings)
        out.append({"geoid": str(a.get("GEOID")), "total": total, "black": black,
                    "lng": lng, "lat": lat})
    return out


def _acs_from_census_api(key: str) -> list[dict]:
    # DP05_0001E (total), DP05_0038PE (%Black) — the ingest_census.py way.
    url = CENSUS_API + "?" + urllib.parse.urlencode({
        "get": "DP05_0001E,DP05_0038PE", "for": "tract:*",
        "in": "state:11 county:001", "key": key,
    })
    raw = _get(url)
    hdr, rows = raw[0], raw[1:]
    pop: dict[str, tuple[int, int]] = {}
    for r in rows:
        rec = dict(zip(hdr, r))
        geoid = "11001" + rec["tract"]
        total = int(float(rec["DP05_0001E"] or 0))
        pct = float(rec["DP05_0038PE"] or 0)
        pop[geoid] = (total, round(total * pct / 100))
    # Geometry/centroids come from the DCGIS ACS tract layer (join on GEOID).
    g = _q(ACS_TRACT_LAYER, {"where": "1=1", "outFields": "GEOID",
                            "returnGeometry": "true", "outSR": "4326", "resultRecordCount": 400})
    out = []
    for f in g["features"]:
        geoid = str(f["attributes"].get("GEOID"))
        rings = (f.get("geometry") or {}).get("rings", [])
        if geoid in pop and rings:
            total, black = pop[geoid]
            lng, lat = _centroid(rings)
            out.append({"geoid": geoid, "total": total, "black": black, "lng": lng, "lat": lat})
    return out


def _centroid(rings: list) -> tuple[float, float]:
    ring = rings[0]  # outer ring; mean of vertices (matches the POC / ingest_census.py)
    return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))
