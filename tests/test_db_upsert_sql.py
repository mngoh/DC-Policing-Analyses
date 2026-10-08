import pytest

from database import db


def test_postgis_upsert_casts_coordinate_parameters_for_geometry():
    sql = db._UPSERT_SQL_POSTGIS

    assert "$12::double precision IS NOT NULL" in sql
    assert "$13::double precision IS NOT NULL" in sql
    assert "ST_MakePoint($13::double precision, $12::double precision)" in sql
