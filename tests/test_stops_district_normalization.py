"""Unit tests for the Stop-Data district normalization + row transform.

The district normalization is the one piece of the stops ingest with real
branching (double-encoded '7'/'7D', 'NA'/null, out-of-range), so it is tested
directly. No database or network is required.
"""
import pytest

from database.stops import STOP_COLUMNS, normalize_district, transform_stop


@pytest.mark.parametrize("raw,expected", [
    # Both encodings collapse to the same integer district.
    ("7", (7, False)),
    ("7D", (7, False)),
    ("1", (1, False)),
    ("1D", (1, False)),
    ("5D", (5, False)),
    # Whitespace / case tolerance.
    (" 7d ", (7, False)),
    ("3D ", (3, False)),
    # Unmappable — flagged, never guessed.
    ("NA", (None, True)),
    ("na", (None, True)),
    (None, (None, True)),
    ("", (None, True)),
    ("   ", (None, True)),
    ("D", (None, True)),      # bare 'D' -> empty after strip
    # Out of the 1-7 range -> flagged.
    ("0", (None, True)),
    ("8", (None, True)),
    ("8D", (None, True)),
    ("10", (None, True)),
    # Garbage -> flagged.
    ("XYZ", (None, True)),
    ("7X", (None, True)),
])
def test_normalize_district(raw, expected):
    assert normalize_district(raw) == expected


def _as_dict(row):
    return dict(zip(STOP_COLUMNS, row))


def test_transform_maps_district_and_preserves_unknown_race():
    row = transform_stop({
        "GLOBALID": "{G1}", "OBJECTID": 12, "CCN_ANONYMIZED": "ccn1",
        "ETHNICITY": "Unknown", "STOP_DISTRICT": "7D", "STOP_TYPE": "Non-ticket Stop",
        "DATETIME": 1704067200000,  # 2024-01-01T00:00:00Z
    }, source_layer=41, ingest_batch=7)
    d = _as_dict(row)
    assert d["ingest_batch"] == 7                # tagged with its snapshot batch
    assert d["global_id"] == "{G1}"
    assert d["ethnicity"] == "Unknown"          # missing-race category kept, never dropped
    assert (d["district"], d["district_unmapped"]) == (7, False)
    assert d["stop_district_raw"] == "7D"        # raw value also retained
    assert d["year"] == 2024
    assert d["source_objectid"] == 12
    assert d["source_layer"] == 41


def test_transform_flags_unmapped_district_and_handles_null_datetime():
    d = _as_dict(transform_stop({"GLOBALID": "{G2}", "STOP_DISTRICT": "NA",
                                 "ETHNICITY": None, "DATETIME": None},
                                source_layer=41, ingest_batch=9))
    assert d["ingest_batch"] == 9
    assert (d["district"], d["district_unmapped"]) == (None, True)
    assert d["stop_district_raw"] == "NA"
    assert d["ethnicity"] is None
    assert d["stop_datetime"] is None and d["year"] is None


def test_transform_skips_row_without_globalid():
    # GLOBALID is half the row identity — a row without it cannot be upserted.
    assert transform_stop({"STOP_DISTRICT": "7", "ETHNICITY": "Black"},
                          source_layer=41, ingest_batch=1) is None
