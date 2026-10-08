"""Tests for the arrests ingest transform and the disparity-cell mapping."""
import pytest

from database.arrests import (
    ARREST_COLUMNS,
    arrest_cells_to_disparity_cells,
    transform_arrest,
)
from database.disparity import representation_by_district


def _attrs(**over):
    base = {
        "GLOBALID": "{ABC-123}", "OBJECTID": 1, "ARREST_NUMBER": 555, "CCN": "123",
        "TYPE": "Adult Arrest", "YEAR": 2024.0, "DATE_": 1700000000000, "HOUR": 13,
        "AGE": 30, "RACE": "BLACK", "ETHNICITY": "NOT HISPANIC", "SEX": "MALE",
        "CATEGORY": "Simple Assault", "DESCRIPTION": "x",
        "ARREST_DISTRICT": "7D", "ARREST_PSA": 704,
        "OFFENSE_DISTRICT": "7D", "OFFENSE_PSA": 704,
        "DEFENDANT_DISTRICT": "6D", "DEFENDANT_PSA": 602,
        "ARREST_LATITUDE": 38.9, "ARREST_LONGITUDE": -76.9,
    }
    base.update(over)
    return base


def test_transform_shape_and_district_normalization():
    row = transform_arrest(_attrs(), ingest_batch=9)
    assert len(row) == len(ARREST_COLUMNS)
    d = dict(zip(ARREST_COLUMNS, row))
    assert d["ingest_batch"] == 9
    assert d["district"] == 7 and d["district_unmapped"] is False   # '7D' normalized
    assert d["year"] == 2024                                        # source YEAR wins
    assert d["arrest_number"] == "555"                              # numerics stored as text
    assert d["arrest_psa"] == "704"


def test_transform_unmappable_district_flagged():
    d = dict(zip(ARREST_COLUMNS, transform_arrest(_attrs(ARREST_DISTRICT="NA"), 1)))
    assert d["district"] is None and d["district_unmapped"] is True


def test_transform_skips_missing_globalid():
    assert transform_arrest(_attrs(GLOBALID=None), 1) is None


def test_cells_mapping_unknown_labels_and_black():
    rows = [
        {"district": 7, "race": "BLACK", "n": 90},
        {"district": 7, "race": "WHITE", "n": 10},
        {"district": 7, "race": "UNK", "n": 5},        # both unknown labels ->
        {"district": 7, "race": "UNKNOWN", "n": 5},    # the engine's 'Unknown'
        {"district": 7, "race": None, "n": 5},
    ]
    cells = arrest_cells_to_disparity_cells(rows)
    unknown_n = sum(c["n"] for c in cells if c["ethnicity"] == "Unknown")
    assert unknown_n == 15

    # And the stops-engine math applies unchanged: 90/100 known-race Black.
    out = representation_by_district(cells, 0.45, {7: {"police_district": "7D",
                                                       "black_pop_share": 0.865}})
    assert out[0]["black_stops"] == 90
    assert out[0]["known_race_stops"] == 100
    assert out[0]["representation_index"] == pytest.approx(2.0)
