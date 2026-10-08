"""Reconciliation test for the district-population denominator aggregation.

The load-time reconciliation (scripts/load_district_population.py) needs real
network data, so this test exercises the same ``aggregate()`` code path OFFLINE
on synthetic tracts + districts (the suite runs with no network and DATABASE_URL
disabled — see tests/conftest.py). It pins the invariant the loader reports:

    sum(district totals) + orphan population == citywide total

i.e. the loaded rows reconcile to the DC total. This is the reconciliation the
findings doc makes ("summed across districts == DC total", 0 orphans), proven on
data whose right answer is known by construction.
"""
from database.district_population import aggregate, build_source_note, CENTROID_CAVEAT


def _square(x0: float, y0: float, size: float = 1.0) -> list:
    """One esri-style outer ring for the square [x0,x0+size] x [y0,y0+size]."""
    return [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]]


# Seven non-overlapping unit squares laid out left to right — a toy stand-in for
# the 7 MPD districts that tile DC.
DISTRICTS = [
    {"district": d, "police_district": f"{d}D", "rings": _square(d * 2.0, 0.0)}
    for d in range(1, 8)
]


def _tract(district_slot: int, total: int, black: int) -> dict:
    """A tract whose centroid sits in the middle of district ``district_slot``."""
    return {"geoid": f"t{district_slot}", "total": total, "black": black,
            "lng": district_slot * 2.0 + 0.5, "lat": 0.5}


def test_reconciles_to_citywide_total_with_no_orphans():
    tracts = [
        _tract(1, 100, 25), _tract(1, 100, 25),   # district 1: 200 pop, 50 Black -> 25%
        _tract(2, 300, 30),                        # district 2: 300 pop, 30 Black -> 10%
        _tract(6, 400, 360),                       # district 6: 400 pop, 360 Black -> 90%
    ]
    rows, citywide, orphans = aggregate(tracts, DISTRICTS)

    assert len(rows) == 7                          # one row per district, always
    assert orphans == []

    by_d = {r["district"]: r for r in rows}
    assert (by_d[1]["total_population"], by_d[1]["black_population"], by_d[1]["pct_black"]) == (200, 50, 25.0)
    assert (by_d[2]["total_population"], by_d[2]["pct_black"]) == (300, 10.0)
    assert (by_d[6]["total_population"], by_d[6]["pct_black"]) == (400, 90.0)
    assert by_d[3]["total_population"] == 0        # empty district -> 0, not dropped

    # The core reconciliation: district sum + orphans == citywide total.
    district_sum = sum(r["total_population"] for r in rows)
    orphan_pop = sum(t["total"] for t in orphans)
    assert district_sum + orphan_pop == citywide["total_population"] == 900
    assert citywide["black_population"] == 440
    assert citywide["pct_black"] == round(100 * 440 / 900, 1)  # 48.9%


def test_orphan_tract_counts_toward_citywide_but_no_district():
    # A tract whose centroid falls outside every district square (far away).
    orphan = {"geoid": "orphan", "total": 500, "black": 100, "lng": 999.0, "lat": 999.0}
    tracts = [_tract(4, 100, 40), orphan]
    rows, citywide, orphans = aggregate(tracts, DISTRICTS)

    assert [t["geoid"] for t in orphans] == ["orphan"]
    assert sum(r["total_population"] for r in rows) == 100      # only the assigned tract
    assert citywide["total_population"] == 600                  # citywide includes the orphan

    # Reconciliation still balances once orphans are accounted for.
    district_sum = sum(r["total_population"] for r in rows)
    orphan_pop = sum(t["total"] for t in orphans)
    assert district_sum + orphan_pop == citywide["total_population"]


def test_source_note_carries_the_centroid_caveat():
    note = build_source_note("DCGIS layer 38 (ACS 2019-2023)")
    assert "DCGIS layer 38" in note
    assert CENTROID_CAVEAT in note                 # the honesty ships with the data
