"""Tests for the camera-vs-officer counts: point-in-polygon and the skew math."""
from datetime import date

import pytest

from database.tickets import (
    assign_district,
    month_layer_url,
    point_in_rings,
    skew_from_counts,
)

# Unit square with a square hole in the middle (even-odd handles holes).
SQUARE = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
SQUARE_WITH_HOLE = SQUARE + [[[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]]


def test_point_in_rings_basic():
    assert point_in_rings(5, 5, SQUARE) is True
    assert point_in_rings(11, 5, SQUARE) is False
    assert point_in_rings(-1, -1, SQUARE) is False


def test_point_in_rings_hole():
    assert point_in_rings(5, 5, SQUARE_WITH_HOLE) is False   # inside the hole
    assert point_in_rings(2, 2, SQUARE_WITH_HOLE) is True


def test_assign_district_null_and_outside():
    polys = {1: SQUARE}
    assert assign_district(5, 5, polys) == 1
    assert assign_district(50, 50, polys) is None
    assert assign_district(None, 5, polys) is None


def test_month_layer_url_pattern():
    assert month_layer_url(2024, 1).endswith("Violations_Moving_2024/MapServer/0")
    assert month_layer_url(2025, 12).endswith("Violations_Moving_2025/MapServer/11")


def _rows():
    m = date(2024, 1, 1)
    return [
        {"month": m, "source": "camera", "district": 2, "n": 800},
        {"month": m, "source": "camera", "district": 7, "n": 200},
        {"month": m, "source": "officer", "district": 2, "n": 20},
        {"month": m, "source": "officer", "district": 7, "n": 80},
        {"month": m, "source": "camera", "district": None, "n": 15},
        {"month": m, "source": "officer", "district": None, "n": 5},
        {"month": m, "source": "other_agency", "district": None, "n": 99},
    ]


def test_skew_math_and_exclusions():
    out = skew_from_counts(_rows(), {2: {"police_district": "2D", "black_pop_share": 0.08}})
    by_d = {r["district"]: r for r in out["districts"]}

    # D2: 80% of camera tickets but 20% of officer tickets -> skew 0.25.
    assert by_d[2]["camera_share"] == pytest.approx(0.80)
    assert by_d[2]["officer_share"] == pytest.approx(0.20)
    assert by_d[2]["skew"] == pytest.approx(0.25)
    # D7: 20% camera, 80% officer -> officers concentrate 4x beyond cameras.
    assert by_d[7]["skew"] == pytest.approx(4.0)

    # Denominators are district-assigned tickets ONLY; exclusions reported.
    assert out["officer_total"] == 100
    assert out["camera_total"] == 1000
    assert out["excluded"] == {"other_agency": 99, "camera_unassigned": 15,
                               "officer_unassigned": 5}
    assert by_d[2]["black_pop_share"] == 0.08


def test_skew_none_when_no_camera_share():
    m = date(2024, 1, 1)
    out = skew_from_counts([{"month": m, "source": "officer", "district": 3, "n": 10}])
    assert out["districts"][0]["skew"] is None
