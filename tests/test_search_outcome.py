"""Tests for the search/hit-rate outcome test — the parser and the pure math.

The hit-rate parser is the load-bearing piece: seizure outcomes are messy,
multi-value, era-varying free text, and a wrong bucket (miss counted as
unrecorded, or vice versa) IS a wrong statistic.
"""
import pytest

from database.disparity import (
    BLACK,
    UNKNOWN_RACE,
    hit_rates_citywide,
    outcome_test_direction,
    parse_seizure,
    search_rates_by_district,
)


# ── parse_seizure: real vocabulary from both layers ───────────────────────────
@pytest.mark.parametrize("text,expected", [
    (None, None),                       # no outcome recorded
    ("NA", None),                       # layer-45 not-applicable marker
    ("", None),
    (" ; ", None),
    ("None", False),                    # recorded miss
    ("None ;", False),                  # layer-45 trailing-separator variant
    ("Gun", True),
    ("Gun ;", True),
    ("Narcotics ;", True),
    ("Drugs/narcotics", True),          # layer-45 vocabulary
    ("Alcohol, None", True),            # something AND None -> something seized
    ("None ; Narcotics ;", True),
    ("Drug Paraphernalia ; Narcotics ;", True),
    ("Gun, Narcotics, Stolen Property", True),
    ("Cell phone(s) or electronic devices, Money", True),
])
def test_parse_seizure(text, expected):
    assert parse_seizure(text) is expected


# ── Search rates ──────────────────────────────────────────────────────────────
def _cell(district, ethnicity, n, pat_down=0, any_search=0, **kw):
    base = {"district": district, "ethnicity": ethnicity, "n": n,
            "consent": 0, "prob_cause": 0, "warrant": 0,
            "pat_down": pat_down, "any_search": any_search,
            "any_search_or_pat_down": max(pat_down, any_search)}
    base.update(kw)
    return base


def test_search_rates_and_ratio():
    cells = [
        _cell(2, BLACK, 1000, pat_down=100, any_search=50),
        _cell(2, "White", 4000, pat_down=80, any_search=40),
    ]
    rows = search_rates_by_district(cells)
    assert len(rows) == 1
    r = rows[0]
    assert r["black"]["pat_down_rate"] == pytest.approx(0.10)
    assert r["non_black_known_race"]["pat_down_rate"] == pytest.approx(0.02)
    assert r["pat_down_rate_ratio"] == pytest.approx(5.0)
    assert r["black"]["search_rate"] == pytest.approx(0.05)


def test_search_rates_exclude_unknown_and_unmapped():
    cells = [
        _cell(1, BLACK, 100, pat_down=10),
        _cell(1, UNKNOWN_RACE, 900, pat_down=900),   # excluded: unknown race
        _cell(None, BLACK, 900, pat_down=900),       # excluded: unmapped district
        _cell(1, None, 900, pat_down=900),           # excluded: SQL NULL race
    ]
    rows = search_rates_by_district(cells)
    assert rows[0]["black"]["stops"] == 100
    assert rows[0]["black"]["pat_down"] == 10


# ── Hit rates ─────────────────────────────────────────────────────────────────
def _hits(ethnicity, search_type, outcome, n):
    return {"ethnicity": ethnicity, "search_type": search_type,
            "outcome": outcome, "n": n}


def test_hit_rates_unrecorded_excluded_from_denominator():
    cells = [
        _hits(BLACK, "pat_down", True, 10),
        _hits(BLACK, "pat_down", False, 40),
        _hits(BLACK, "pat_down", None, 50),          # unrecorded: not a miss
        _hits("White", "pat_down", True, 10),
        _hits("White", "pat_down", False, 10),
    ]
    g = hit_rates_citywide(cells)
    b = g["black"]["pat_down"]
    assert b["searched"] == 100
    assert b["recorded"] == 50
    assert b["unrecorded"] == 50
    assert b["hit_rate"] == pytest.approx(10 / 50)   # NOT 10/100
    assert g["non_black_known_race"]["pat_down"]["hit_rate"] == pytest.approx(0.5)


def test_hit_rates_combined_sums_types_and_ignores_prob_cause():
    cells = [
        _hits(BLACK, "consent", True, 5),
        _hits(BLACK, "warrant", False, 5),
        _hits(BLACK, "prob_cause", True, 999),       # no outcome field in source
        _hits(BLACK, "pat_down", True, 5),
    ]
    combined = hit_rates_citywide(cells)["black"]["combined"]
    assert combined["searched"] == 15                # 999 never leaks in
    assert combined["hit_rate"] == pytest.approx(10 / 15)


def test_unknown_race_excluded_from_hit_rates():
    cells = [
        _hits(UNKNOWN_RACE, "pat_down", True, 100),
        _hits(None, "pat_down", True, 100),
        _hits(BLACK, "pat_down", False, 10),
    ]
    g = hit_rates_citywide(cells)
    assert g["black"]["pat_down"]["searched"] == 10
    assert g["non_black_known_race"]["pat_down"]["searched"] == 0


# ── Direction: significance-gated ─────────────────────────────────────────────
def _bucket(seized, recorded):
    return {"seized": seized, "recorded": recorded,
            "hit_rate": seized / recorded if recorded else None}


def test_direction_significant_gap_asserted():
    d = outcome_test_direction(_bucket(1000, 10000), _bucket(3000, 10000))  # 10% vs 30%
    assert d["direction"] == "lower" and d["significant"] is True
    d = outcome_test_direction(_bucket(3000, 10000), _bucket(1000, 10000))
    assert d["direction"] == "higher" and d["significant"] is True


def test_direction_noise_never_asserted():
    """The live layer-41 shape: 13.9% vs 12.5% on a 1k-search comparison group
    is ~1.3 SE — must read 'about the same', not 'higher'."""
    d = outcome_test_direction(_bucket(1861, 13406), _bucket(135, 1081))
    assert d["direction"] == "about the same"
    assert d["significant"] is False
    assert abs(d["z"]) < 1.96


def test_direction_not_comparable_on_empty_group():
    d = outcome_test_direction(_bucket(0, 0), _bucket(10, 100))
    assert d["direction"] == "not comparable"


# ── Pure render ───────────────────────────────────────────────────────────────
def _outcome_test_dict(direction=None):
    hit_cells = [
        _hits(BLACK, "pat_down", True, 10), _hits(BLACK, "pat_down", False, 40),
        _hits("White", "pat_down", True, 15), _hits("White", "pat_down", False, 15),
    ]
    return {
        "citywide_hit_rates": hit_rates_citywide(hit_cells),
        "direction": direction or {"direction": "lower", "z": -5.0, "significant": True},
        "districts": search_rates_by_district([
            _cell(2, BLACK, 1000, pat_down=100, any_search=50),
            _cell(2, "White", 4000, pat_down=80, any_search=40),
        ]),
        "exclusions": {},
    }
