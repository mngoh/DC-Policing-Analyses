"""Tests for the enforcement-vs-crime baseline math."""
import pytest

from database.crime_baseline import group_ratio, shares, skew_table

META = {
    2: {"police_district": "2D", "black_pop_share": 0.08},
    7: {"police_district": "7D", "black_pop_share": 0.865},
}


def test_shares_sum_to_one():
    s = shares({1: 25, 2: 75})
    assert s[1] == pytest.approx(0.25) and s[2] == pytest.approx(0.75)


def test_skew_is_share_over_crime_share():
    # D2: 20% of crime but 40% of stops -> skew 2.0; D7 mirrors at 0.75.
    table = skew_table({2: 200, 7: 800}, {"stops": {2: 400, 7: 600}}, META)
    by_d = {r["district"]: r for r in table}
    assert by_d[2]["stops_skew"] == pytest.approx(2.0)
    assert by_d[7]["stops_skew"] == pytest.approx(0.75)
    assert by_d[2]["crime_share"] == pytest.approx(0.2)
    assert by_d[2]["black_pop_share"] == 0.08


def test_skew_matches_crime_means_one():
    table = skew_table({2: 300, 7: 700}, {"stops": {2: 300, 7: 700}}, META)
    for r in table:
        assert r["stops_skew"] == pytest.approx(1.0)


def test_group_ratio_reads_direction_from_data():
    # Majority-Black D7: 700 crimes, 350 stops (0.5/crime).
    # Rest (D2): 300 crimes, 600 stops (2.0/crime). Ratio = 0.25.
    g = group_ratio({2: 300, 7: 700}, {2: 600, 7: 350}, META)
    assert g["majority_black_districts"] == [7]
    assert g["majority_black"]["per_crime"] == pytest.approx(0.5)
    assert g["rest"]["per_crime"] == pytest.approx(2.0)
    assert g["ratio"] == pytest.approx(0.25)


def test_group_ratio_even_enforcement_is_one():
    g = group_ratio({2: 100, 7: 400}, {2: 50, 7: 200}, META)
    assert g["ratio"] == pytest.approx(1.0)


def test_frisk_hits_geography_split_and_gate():
    from database.crime_baseline import frisk_hits_by_geography
    rows = (
        [{"district": 7, "outcome": True, "n": 150},
         {"district": 7, "outcome": False, "n": 850},
         {"district": 7, "outcome": None, "n": 30},      # unrecorded: reported
         {"district": 2, "outcome": True, "n": 60},
         {"district": 2, "outcome": False, "n": 940},
         {"district": None, "outcome": True, "n": 99}]   # unmapped: excluded
    )
    out = frisk_hits_by_geography(rows, {5, 6, 7})
    assert out["majority_black"]["hit_rate"] == pytest.approx(0.15)
    assert out["majority_black"]["unrecorded"] == 30
    assert out["rest"]["hit_rate"] == pytest.approx(0.06)
    assert out["direction"]["direction"] == "higher"
    assert out["direction"]["significant"] is True


def test_missing_numerator_district_is_zero_not_missing():
    table = skew_table({2: 100, 7: 100}, {"stops": {2: 100}}, META)
    d7 = next(r for r in table if r["district"] == 7)
    assert d7["stops"] == 0
    assert d7["stops_share"] == pytest.approx(0.0)
    assert d7["stops_skew"] == pytest.approx(0.0)
