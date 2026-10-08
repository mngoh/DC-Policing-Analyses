"""Shared ArcGIS layer constants for DC MPD crime data."""

_BASE = "https://maps2.dcgis.dc.gov/dcgis/rest/services/FEEDS/MPD/FeatureServer/{layer}/query"

ARCGIS_RECENT: str = _BASE.format(layer=39)

YEAR_LAYERS: dict[int, str] = {
    2026: _BASE.format(layer=41),
    2025: _BASE.format(layer=7),
    2024: _BASE.format(layer=6),
    2023: _BASE.format(layer=5),
    2022: _BASE.format(layer=4),
    2021: _BASE.format(layer=3),
    2020: _BASE.format(layer=2),
    2019: _BASE.format(layer=1),
    2018: _BASE.format(layer=0),
    2017: _BASE.format(layer=38),
    2016: _BASE.format(layer=26),
    2015: _BASE.format(layer=27),
    2014: _BASE.format(layer=40),
    2013: _BASE.format(layer=10),
    2012: _BASE.format(layer=11),
    2011: _BASE.format(layer=35),
    2010: _BASE.format(layer=34),
    2009: _BASE.format(layer=33),
    2008: _BASE.format(layer=32),
}
