"""Tests for the veil-of-darkness engine: solar events, MH math, classification.

Solar times are checked against published values for Washington, DC (USNO). The
strata are 30 minutes wide, so a few minutes of algorithmic error is harmless;
the tests allow 8 minutes.
"""
from datetime import date, datetime, timedelta

import pytest

from database.veil import (
    dst_season,
    mantel_haenszel,
    sun_events_utc,
    us_dst_transitions,
    veil_from_rows,
)


def _close(dt: datetime, expected: datetime, minutes: int = 8) -> bool:
    return abs((dt - expected).total_seconds()) <= minutes * 60


# ── Solar events vs published DC times ────────────────────────────────────────
def test_summer_solstice_sunset_and_dusk():
    # 2024-06-20 DC: sunset 8:37 PM EDT = 00:37 UTC (next day); civil dusk
    # ~9:09 PM EDT = 01:09 UTC.
    sunset, dusk = sun_events_utc(date(2024, 6, 20))
    assert _close(sunset, datetime(2024, 6, 21, 0, 37))
    assert _close(dusk, datetime(2024, 6, 21, 1, 9))


def test_winter_solstice_sunset_and_dusk():
    # 2024-12-21 DC: sunset 4:50 PM EST = 21:50 UTC; civil dusk ~5:20 PM EST.
    sunset, dusk = sun_events_utc(date(2024, 12, 21))
    assert _close(sunset, datetime(2024, 12, 21, 21, 50))
    assert _close(dusk, datetime(2024, 12, 21, 22, 20))


def test_dusk_always_after_sunset():
    for d in (date(2023, 3, 12), date(2023, 11, 5), date(2025, 7, 4)):
        sunset, dusk = sun_events_utc(d)
        assert timedelta(0) < dusk - sunset < timedelta(minutes=45)


# ── Mantel-Haenszel ───────────────────────────────────────────────────────────
def test_mh_single_stratum_matches_plain_odds_ratio():
    out = mantel_haenszel([{"day_black": 100, "day_nonblack": 900,
                            "dark_black": 50, "dark_nonblack": 950}])
    assert out["or"] == pytest.approx((100 * 950) / (900 * 50))
    assert out["significant"] is True and out["z"] > 1.96


def test_mh_null_effect_not_significant():
    out = mantel_haenszel([{"day_black": 50, "day_nonblack": 50,
                            "dark_black": 50, "dark_nonblack": 50}])
    assert out["or"] == pytest.approx(1.0)
    assert out["significant"] is False


def test_mh_confounded_strata_adjusted():
    """Two strata with identical within-stratum odds (OR=1) but different
    lighting mixes must pool to OR=1: the stratification is doing its job."""
    out = mantel_haenszel([
        {"day_black": 80, "day_nonblack": 20, "dark_black": 8, "dark_nonblack": 2},
        {"day_black": 2, "day_nonblack": 8, "dark_black": 20, "dark_nonblack": 80},
    ])
    assert out["or"] == pytest.approx(1.0)
    assert out["significant"] is False


def test_mh_empty_arm_strata_skipped():
    out = mantel_haenszel([
        {"day_black": 10, "day_nonblack": 10, "dark_black": 0, "dark_nonblack": 0},
    ])
    assert out["or"] is None and out["significant"] is False


# ── Classification + window from rows ─────────────────────────────────────────
def _row(local_date, local_minute, utc_dt, is_black, ticket_only=False,
         offset_minutes=-300, district=None):
    return {"local_date": local_date, "local_minute": local_minute,
            "utc_dt": utc_dt, "offset_minutes": offset_minutes,
            "is_black": is_black, "ticket_only": ticket_only,
            "district": district}


# Two-season fixture: the window only exists ACROSS seasons. December's early
# sunset (~4:50 PM) sets the window start; June's late dusk (~9:09 PM) sets the
# end. 6:00 PM local is then daylight on the June date and full dark on the
# December date: the exact contrast the design exploits.
WINTER, SUMMER = date(2024, 12, 21), date(2024, 6, 20)
SIX_PM = 18 * 60


def test_rows_classified_light_dark_ambiguous():
    winter_sunset, _ = sun_events_utc(WINTER)
    rows = [
        # June, 6:00 PM local (22:00 UTC, EDT): daylight, inside window.
        _row(SUMMER, SIX_PM, datetime(2024, 6, 20, 22, 0), True, offset_minutes=-240),
        _row(SUMMER, SIX_PM, datetime(2024, 6, 20, 22, 0), False, offset_minutes=-240),
        # December, 6:00 PM local (23:00 UTC, EST): after civil dusk = dark.
        _row(WINTER, SIX_PM, datetime(2024, 12, 21, 23, 0), False),
        # December, 10 minutes after sunset: ambiguous twilight, excluded+counted.
        _row(WINTER,
             (winter_sunset.hour * 60 + winter_sunset.minute - 300 + 10) % 1440,
             winter_sunset + timedelta(minutes=10), True),
    ]
    out = veil_from_rows(rows)
    assert out["all"]["ambiguous_excluded"] == 1
    assert out["all"]["daylight"]["total"] == 2
    assert out["all"]["daylight"]["black"] == 1
    assert out["all"]["dark"]["total"] == 1
    assert "to" in out["window"]["label"]


def test_district_stratification_removes_deployment_confound():
    """Construct a pure deployment confound: within EACH district, lighting
    makes no difference to who is stopped, but one district is mostly-Black and
    mostly-dark while the other is mostly-white and mostly-daylight. The
    time-only OR then departs from 1 (spuriously); the district-stratified OR
    must sit at exactly 1."""
    day_utc, day_kw = datetime(2024, 6, 20, 22, 0), {"offset_minutes": -240}
    dark_utc = datetime(2024, 12, 21, 23, 0)

    rows = []
    # District 7: 90% Black regardless of lighting; observed mostly after dark.
    for utc, d, n_scale in ((dark_utc, WINTER, 9), (day_utc, SUMMER, 1)):
        kw = day_kw if d == SUMMER else {}
        rows += [_row(d, SIX_PM, utc, True, district=7, **kw)] * (18 * n_scale)
        rows += [_row(d, SIX_PM, utc, False, district=7, **kw)] * (2 * n_scale)
    # District 2: 10% Black regardless of lighting; observed mostly in daylight.
    for utc, d, n_scale in ((dark_utc, WINTER, 1), (day_utc, SUMMER, 9)):
        kw = day_kw if d == SUMMER else {}
        rows += [_row(d, SIX_PM, utc, True, district=2, **kw)] * (2 * n_scale)
        rows += [_row(d, SIX_PM, utc, False, district=2, **kw)] * (18 * n_scale)

    out = veil_from_rows(rows)
    time_only = out["all"]["mh"]["or"]
    stratified = out["all"]["mh_time_district"]["or"]
    assert time_only < 0.9                      # the spurious citywide signal
    assert stratified == pytest.approx(1.0)     # dissolves within districts
    assert out["all"]["district_unmapped_excluded"] == 0


def test_unmapped_district_rows_counted_not_stratified():
    rows = [
        _row(WINTER, SIX_PM, datetime(2024, 12, 21, 23, 0), False, district=None),
        _row(SUMMER, SIX_PM, datetime(2024, 6, 20, 22, 0), True,
             offset_minutes=-240, district=3),
    ]
    out = veil_from_rows(rows)
    assert out["all"]["district_unmapped_excluded"] == 1
    assert out["all"]["daylight"]["total"] + out["all"]["dark"]["total"] == 2


def test_us_dst_transitions_known_years():
    # Published US transition dates (second Sunday of March, first of November).
    assert us_dst_transitions(2024) == (date(2024, 3, 10), date(2024, 11, 3))
    assert us_dst_transitions(2026) == (date(2026, 3, 8), date(2026, 11, 1))
    assert us_dst_transitions(2019) == (date(2019, 3, 10), date(2019, 11, 3))


def test_dst_season_windows():
    assert dst_season(date(2024, 3, 20)) == "spring"     # 10 days after 3/10
    assert dst_season(date(2024, 2, 9)) == "spring"      # exactly 30 days before
    assert dst_season(date(2024, 10, 20)) == "fall"      # 14 days before 11/3
    assert dst_season(date(2024, 12, 4)) is None         # 31 days after 11/3
    assert dst_season(date(2024, 6, 20)) is None         # midsummer
    assert dst_season(date(2024, 4, 10)) is None         # 31 days after 3/10


# March-transition fixture: 6:45 PM local is dark on Mar 1 (EST, civil dusk
# ~6:26 PM) and daylight on Mar 20 (EDT, sunset ~7:14 PM). Both dates are
# within 30 days of the 2024-03-10 transition, so both feed the dst block;
# season is near-constant and the clock change supplies the lighting contrast.
PRE_DST, POST_DST = date(2024, 3, 1), date(2024, 3, 20)
QUARTER_TO_SEVEN = 18 * 60 + 45


def _march_rows():
    rows = []
    # Mar 20, 6:45 PM EDT = 22:45 UTC: daylight.
    rows += [_row(POST_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 20, 22, 45),
                  True, offset_minutes=-240),
             _row(POST_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 20, 22, 45),
                  False, offset_minutes=-240)]
    # Mar 1, 6:45 PM EST = 23:45 UTC: after civil dusk = dark.
    rows += [_row(PRE_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 1, 23, 45), False)]
    return rows


def test_dst_block_mirrors_transition_window_rows():
    # A midsummer daylight row is in the full-year test but NOT the dst block.
    rows = _march_rows() + [
        _row(SUMMER, QUARTER_TO_SEVEN, datetime(2024, 6, 20, 22, 45), True,
             offset_minutes=-240),
    ]
    out = veil_from_rows(rows)
    a, d = out["all"], out["all"]["dst"]
    assert a["daylight"]["total"] == 3 and a["dark"]["total"] == 1
    assert d["daylight"]["total"] == 2 and d["dark"]["total"] == 1
    assert d["daylight"]["black"] == 1 and d["dark"]["black"] == 0
    assert d["days"] == 30


def test_dst_strata_keep_seasons_apart():
    """Spring daylight rows against fall dark rows at the same clock time must
    NOT form a comparison: with season in the stratum key, each stratum has an
    empty arm and the estimator abstains (or=None). Naive pooling would have
    manufactured an odds ratio from pure seasonal contrast."""
    fall_dark = date(2024, 11, 20)                       # 17 days after 11/3, EST
    rows = [
        _row(POST_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 20, 22, 45), True,
             offset_minutes=-240),
        _row(POST_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 20, 22, 45), False,
             offset_minutes=-240),
        _row(fall_dark, QUARTER_TO_SEVEN, datetime(2024, 11, 20, 23, 45), True),
        _row(fall_dark, QUARTER_TO_SEVEN, datetime(2024, 11, 20, 23, 45), False),
    ]
    out = veil_from_rows(rows)
    assert out["all"]["dst"]["mh"]["or"] is None
    assert out["all"]["dst"]["mh"]["significant"] is False


def test_dst_district_stratified_and_unmapped_counted():
    rows = [
        _row(POST_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 20, 22, 45), True,
             offset_minutes=-240, district=3),
        _row(PRE_DST, QUARTER_TO_SEVEN, datetime(2024, 3, 1, 23, 45), False,
             district=None),
    ]
    out = veil_from_rows(rows)
    assert out["all"]["dst"]["district_unmapped_excluded"] == 1
    assert out["all"]["district_unmapped_excluded"] == 1


def test_ticket_only_subset_is_a_subset():
    rows = [
        # Winter date pins the window start below 6:00 PM; both June rows are
        # in-window daylight.
        _row(WINTER, SIX_PM, datetime(2024, 12, 21, 23, 0), False),
        _row(SUMMER, SIX_PM, datetime(2024, 6, 20, 22, 0), True,
             ticket_only=True, offset_minutes=-240),
        _row(SUMMER, SIX_PM, datetime(2024, 6, 20, 22, 0), False,
             ticket_only=False, offset_minutes=-240),
    ]
    out = veil_from_rows(rows)
    assert out["all"]["daylight"]["total"] == 2
    assert out["ticket_only"]["daylight"]["total"] == 1
    assert out["ticket_only"]["daylight"]["black"] == 1
