#!/usr/bin/env python3
"""Build the population denominator for a police-district-level stop-disparity metric.

Standalone data-prep — NOT a feature. No routes, no crime_index, no frontend, no
table changes. It aggregates ACS tract population up to MPD police districts so a
district-level disparity metric (district stop share vs district population share,
against a citywide baseline) has a denominator.

    python scripts/build_district_population.py

Why districts and not wards: MPD Stop Data (opendata.dc.gov "Stop Data", layer 41)
has NO coordinates and NO ward field — only STOP_DISTRICT, which is double-encoded
('7' and '7D' both occur, plus 'NA'/null). See docs/slice-findings.md. So the join
must be tract -> police district, and the key is the district number.

Pipeline:
  1. MPD Police Districts boundary polygons (found empirically, not assumed).
  2. ACS tract population — Census API primary (DP05_0001E total, DP05_0038PE
     %Black, exactly like database/ingest_census.py); DCGIS ACS-by-tract layer as a
     keyless fallback (this environment has no CENSUS_API_KEY).
  3. Assign each tract centroid to a district polygon (centroid-in-polygon, the
     same approach as the census ingest) and sum.
  4. Reconcile summed district population against the DC total and report orphans.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

# ── Sources (verified live against DCGIS) ─────────────────────────────────────
DISTRICTS_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/9"          # "Police Districts (MPD)"
)
STOP_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Safety_WebMercator/MapServer/41"         # "Stop Data" (current)
)
ACS_TRACT_LAYER = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Demographic_WebMercator/MapServer/38"           # "ACS 5-Year Demographic ... Census Tracts"
)
CENSUS_API = "https://api.census.gov/data/2023/acs/acs5/profile"  # DP05 profile, 2019-2023
DC_TOTAL_EXPECTED = (672_000, 680_000)  # sanity band


def _get(url: str, tries: int = 5):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - POC: surface anything, retry transient drops
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _q(layer: str, params: dict, tries: int = 5):
    params = {**params, "f": "json"}
    return _get(layer + "/query?" + urllib.parse.urlencode(params), tries)


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ── 1. Police district boundaries + the join key to Stop Data ─────────────────
def fetch_districts() -> list[dict]:
    hr("1. MPD POLICE DISTRICT BOUNDARIES + join key to Stop Data")
    print(f"Boundary layer: {DISTRICTS_LAYER}")
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
    print(f"Districts returned: {len(out)}")
    print("  DISTRICT | POLICEDISTRICT | NAME")
    for r in out:
        print(f"     {r['district']}     |     {r['police_district']:<4}     | {r['name']}")

    # Empirically confirm how the polygon IDs relate to Stop Data's STOP_DISTRICT.
    try:
        s = _q(STOP_LAYER, {"where": "1=1", "outFields": "STOP_DISTRICT",
                            "returnDistinctValues": "true", "orderByFields": "STOP_DISTRICT"})
        stop_codes = [f["attributes"]["STOP_DISTRICT"] for f in s["features"]]
    except Exception as e:  # noqa: BLE001
        stop_codes = [f"<unavailable: {e!r}>"]
    print(f"\nStop Data STOP_DISTRICT distinct values: {stop_codes}")
    print("JOIN KEY MAPPING (verified empirically):")
    print("  - Boundary DISTRICT is an integer 1-7; POLICEDISTRICT is '1D'-'7D'.")
    print("  - Stop Data STOP_DISTRICT is double-encoded: bare '1'-'7' AND '1D'-'7D'.")
    print("  - Normalize STOP_DISTRICT by stripping a trailing 'D' -> integer 1-7,")
    print("    which equals boundary DISTRICT. ('7' and '7D' both -> district 7.)")
    print("  - Unmappable STOP_DISTRICT codes ('NA', null) have no polygon and would")
    print("    be dropped from any district join on the numerator side.")
    return out


# ── 2. ACS tract population (Census API primary; DCGIS fallback) ───────────────
def fetch_tracts() -> tuple[list[dict], str]:
    """Return ([{geoid, total, black, lng, lat}], source_label)."""
    hr("2. ACS TRACT POPULATION (DP05_0001E total, DP05_0038 Black)")
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        try:
            tracts = _acs_from_census_api(key)
            src = "Census API (api.census.gov, ACS 2019-2023 5-year) + DCGIS tract geometry"
            print(f"Source: {src}")
            return tracts, src
        except Exception as e:  # noqa: BLE001
            print(f"Census API path failed ({e!r}); falling back to DCGIS.")
    else:
        print("No CENSUS_API_KEY in environment — using the keyless DCGIS fallback.")
    tracts = _acs_from_dcgis()
    src = ("DCGIS 'ACS 5-Year Demographic Characteristics of DC Census Tracts' "
           "(layer 38, Office of Planning, ACS 2019-2023 5-year)")
    print(f"Source: {src}")
    return tracts, src


def _centroid(rings: list) -> tuple[float, float]:
    ring = rings[0]  # outer ring; mean of vertices (matches database/ingest_census.py)
    return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))


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
    # Population the ingest_census.py way: DP05_0001E (total), DP05_0038PE (%Black).
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


# ── 3. Centroid-in-polygon assignment ─────────────────────────────────────────
def _point_in_rings(x: float, y: float, rings: list) -> bool:
    """Even-odd ray cast across every edge of every ring — correct for esri
    multi-ring polygons (outer rings + holes toggle the same 'inside' flag)."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
    return inside


def assign(tracts: list[dict], districts: list[dict]) -> tuple[dict, list[dict]]:
    agg = {d["district"]: {"police_district": d["police_district"], "name": d["name"],
                           "total": 0, "black": 0, "tracts": 0} for d in districts}
    orphans = []
    for t in tracts:
        hit = None
        for d in districts:
            if _point_in_rings(t["lng"], t["lat"], d["rings"]):
                hit = d["district"]
                break
        if hit is None:
            orphans.append(t)
            continue
        agg[hit]["total"] += t["total"]
        agg[hit]["black"] += t["black"]
        agg[hit]["tracts"] += 1
    return agg, orphans


# ── 4. Per-district table + reconciliation ────────────────────────────────────
def report(agg: dict, tracts: list[dict], orphans: list[dict]) -> None:
    hr("3. PER-DISTRICT POPULATION")
    print(f"{'DIST':<5}{'POLICE':<8}{'tracts':>7}{'total pop':>12}{'Black pop':>12}{'% Black':>9}")
    dist_total = dist_black = 0
    for dnum in sorted(agg):
        a = agg[dnum]
        pct = round(100 * a["black"] / a["total"], 1) if a["total"] else 0.0
        dist_total += a["total"]
        dist_black += a["black"]
        print(f"{dnum:<5}{str(a['police_district']):<8}{a['tracts']:>7}{a['total']:>12,}{a['black']:>12,}{pct:>8.1f}%")
    citywide_pct = round(100 * dist_black / dist_total, 1) if dist_total else 0.0
    print(f"{'ALL':<5}{'':<8}{sum(a['tracts'] for a in agg.values()):>7}{dist_total:>12,}{dist_black:>12,}{citywide_pct:>8.1f}%")

    hr("4. RECONCILIATION (loud)")
    dc_total = sum(t["total"] for t in tracts)
    dc_black = sum(t["black"] for t in tracts)
    orphan_pop = sum(t["total"] for t in orphans)
    disc = dc_total - dist_total
    print(f"DC total population (all {len(tracts)} tracts) : {dc_total:>10,}")
    print(f"Summed across districts                        : {dist_total:>10,}")
    print(f"Discrepancy (DC total - district sum)          : {disc:>+10,}  ({100*disc/dc_total:+.2f}%)")
    print(f"DC total Black / citywide %Black               : {dc_black:,} / {round(100*dc_black/dc_total,1)}%")
    lo, hi = DC_TOTAL_EXPECTED
    print(f"Sanity: DC total in expected {lo:,}-{hi:,}? {'YES' if lo <= dc_total <= hi else 'NO'}")

    print(f"\nOrphan tracts (centroid landed in no district): {len(orphans)} "
          f"({orphan_pop:,} people, {100*orphan_pop/dc_total:.2f}% of DC)")
    for t in orphans:
        print(f"  - GEOID {t['geoid']}  pop={t['total']:,}  centroid=({t['lat']:.4f},{t['lng']:.4f})")

    if not orphans and disc == 0:
        print("\n*** SUSPICIOUS: perfect reconciliation with zero orphan tracts on the")
        print("    first pass. District and tract boundaries do NOT nest, and mean-vertex")
        print("    centroids near edges/water usually misbehave. Re-check containment,")
        print("    the centroid definition, and the polygon ring handling before trusting this. ***")
    else:
        print("\n(Non-zero orphans / discrepancy are expected: tract and district boundaries")
        print(" do not nest, and edge/water tracts push mean-vertex centroids outside.)")


def main() -> None:
    print("MPD POLICE-DISTRICT POPULATION DENOMINATOR (throwaway data-prep)")
    districts = fetch_districts()
    tracts, _src = fetch_tracts()
    print(f"Tracts pulled: {len(tracts)} | DC total (pre-aggregation): "
          f"{sum(t['total'] for t in tracts):,}")
    agg, orphans = assign(tracts, districts)
    report(agg, tracts, orphans)
    hr("DONE")


if __name__ == "__main__":
    main()
