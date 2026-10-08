"""The bulk-ingest year window must cover the entire calendar year.

ArcGIS DATE literals are midnight timestamps, so an inclusive
`<= DATE 'YYYY-12-31'` silently drops any Dec-31 record carrying a time of
day. This pins the end-exclusive window in database/ingest.py by evaluating
the boundary instants against the parsed window (a semantic check, not a
string match).
"""
import re
from datetime import datetime

from database.ingest import year_where


def _window(where: str):
    m = re.fullmatch(
        r"REPORT_DAT (>=|>) DATE '(\d{4}-\d{2}-\d{2})' AND REPORT_DAT (<=|<) DATE '(\d{4}-\d{2}-\d{2})'",
        where,
    )
    assert m, f"unexpected where shape: {where!r}"
    lo_op, lo, hi_op, hi = m.groups()
    return lo_op, datetime.fromisoformat(lo), hi_op, datetime.fromisoformat(hi)


def _contains(where: str, instant: datetime) -> bool:
    lo_op, lo, hi_op, hi = _window(where)
    above = instant >= lo if lo_op == ">=" else instant > lo
    below = instant < hi if hi_op == "<" else instant <= hi
    return above and below


def test_year_window_includes_dec_31_with_time_of_day():
    where = year_where(2023)
    assert _contains(where, datetime(2023, 1, 1, 0, 0, 0)), "Jan 1 00:00 must be included"
    assert _contains(where, datetime(2023, 12, 31, 9, 30, 0)), f"Dec 31 daytime report excluded by {where!r}"
    assert _contains(where, datetime(2023, 12, 31, 23, 59, 59)), f"Dec 31 23:59:59 excluded by {where!r}"


def test_year_window_excludes_the_next_year():
    where = year_where(2023)
    assert not _contains(where, datetime(2024, 1, 1, 0, 0, 0))
