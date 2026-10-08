"""Veil-of-darkness test (Grogger & Ridgeway 2006) on MPD stop data.

THE DESIGN. If officers can see (and act on) a person's race before deciding to
stop them, then stops made in DAYLIGHT should skew Blacker than stops made in
DARKNESS at the same clock time. Daylight-saving transitions supply the natural
experiment: 7:00 PM is daylight in June and pitch dark in December, so the same
clock hour is observed under both lighting conditions. The comparison is
restricted to the INTER-TWILIGHT WINDOW (from the year's earliest sunset to its
latest civil dusk), the clock-time range that can be either light or dark, and
is stratified by 30-minute clock bins so lighting is compared within the same
time of day, never across it.

Estimator: Mantel-Haenszel odds ratio across clock-time strata (each stratum a
2x2 table: daylight/dark x Black/non-Black), with the Robins-Breslow-Greenland
variance for the CI and z-test. OR > 1 means daylight stops skew Blacker, the
signature of visibility-based selection; the page asserts a direction ONLY when
the test is significant (same gate as the outcome test).

Sun times: NOAA/Almanac solar position algorithm, pure Python, accurate to a
few minutes (unit-tested against published DC sunset/dusk times, which is well
inside the 30-minute strata). A stop between sunset and civil dusk (sun 0 to 6
degrees below the horizon) is AMBIGUOUS lighting: excluded and counted, never
guessed. Timezone handling never touches the OS tz database: local timestamps
come from Postgres (AT TIME ZONE), and each date's UTC offset is derived from
the rows themselves.

SEASONAL ROBUSTNESS (the transition-window check). Pooling the full year has a
known confound: at a given clock time, the daylight arm is mostly summer and
the dark arm mostly winter, and who is out on the street differs by season for
reasons unrelated to visibility. The standard mitigation (Ridgeway's follow-up
practice) is to restrict to stops within DST_WINDOW_DAYS of a daylight-saving
transition, where season is held nearly constant and the clock itself supplies
the lighting change. Every subset therefore carries a second pair of estimates
computed on that restricted sample, with strata of clock bin x transition
season (spring/fall) - and clock bin x season x district for the stratified
variant - so a spring daylight stop is never compared against a fall dark stop.

Honest limitations, stated on the surface: MPD stop rows do not carry a clean
vehicle/pedestrian split, so the test runs on all stops (visibility argument
holds for street stops generally) with the 'Ticket Only' subset reported as
the closest proxy for classic traffic stops; grain is stops of persons; both
eras are pooled (2019-2025).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional

from database.disparity import BLACK, _is_known_race, resolve_current_batch

DC_LAT = 38.9072
DC_LON = -77.0369
ZENITH_SUNSET = 90.833          # official sunset (refraction-corrected)
ZENITH_CIVIL_DUSK = 96.0        # civil twilight end: sun 6 degrees below horizon
STRATUM_MINUTES = 30
DST_WINDOW_DAYS = 30            # transition-window check: stops within this many
                                # days of a DST transition (season ~constant)

VEIL_DEFINITION = (
    "Within the inter-twilight window (earliest sunset to latest civil dusk, "
    "local clock time), compare the odds that a stopped person is Black in "
    "DAYLIGHT vs in DARKNESS at the same clock time (30-minute strata), pooled "
    "with a Mantel-Haenszel odds ratio. OR > 1 means daylight stops skew "
    "Blacker, consistent with officers selecting on what they can see. Stops "
    "between sunset and civil dusk are ambiguous lighting: excluded and "
    "reported. Unknown-race stops excluded and reported. All stops are "
    "included (MPD data has no vehicle/pedestrian flag); the Ticket Only "
    "subset is reported as the closest proxy for classic traffic stops. "
    "Seasonal robustness: each estimate is recomputed on stops within "
    f"{DST_WINDOW_DAYS} days of a daylight-saving transition, where season is "
    "held nearly constant and the clock change itself supplies the lighting "
    "contrast; those strata are clock bin x transition season, so spring and "
    "fall rows are never compared with each other."
)


# ── NOAA/Almanac solar events (pure) ─────────────────────────────────────────
def _sun_set_utc_hours(day_of_year: int, lat: float, lon: float, zenith: float) -> Optional[float]:
    """UT hour (0-24) of the sun crossing ``zenith`` on its way DOWN.

    Classic 'Almanac for Computers' algorithm. Returns None in polar edge cases
    (never for DC latitudes).
    """
    d2r, r2d = math.pi / 180.0, 180.0 / math.pi
    lng_hour = lon / 15.0
    t = day_of_year + ((18.0 - lng_hour) / 24.0)
    m = (0.9856 * t) - 3.289
    l = m + (1.916 * math.sin(m * d2r)) + (0.020 * math.sin(2 * m * d2r)) + 282.634
    l %= 360.0
    ra = r2d * math.atan(0.91764 * math.tan(l * d2r))
    ra %= 360.0
    # Right ascension into the same quadrant as L, then into hours.
    ra += (math.floor(l / 90.0) * 90.0) - (math.floor(ra / 90.0) * 90.0)
    ra /= 15.0
    sin_dec = 0.39782 * math.sin(l * d2r)
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = ((math.cos(zenith * d2r) - (sin_dec * math.sin(lat * d2r)))
             / (cos_dec * math.cos(lat * d2r)))
    if cos_h < -1.0 or cos_h > 1.0:
        return None
    h = r2d * math.acos(cos_h) / 15.0            # setting event
    ut = h + ra - (0.06571 * t) - 6.622 - lng_hour
    return ut % 24.0


def sun_events_utc(local_date, lat: float = DC_LAT, lon: float = DC_LON) -> tuple:
    """(sunset_utc, civil_dusk_utc) as aware-naive UTC datetimes for the LOCAL
    calendar date. For western longitudes an evening event can land after
    midnight UTC; a small UT hour therefore belongs to the next UTC day."""
    from datetime import datetime, timedelta

    doy = local_date.timetuple().tm_yday
    out = []
    for zenith in (ZENITH_SUNSET, ZENITH_CIVIL_DUSK):
        ut = _sun_set_utc_hours(doy, lat, lon, zenith)
        base = datetime(local_date.year, local_date.month, local_date.day)
        dt = base + timedelta(hours=ut)
        if ut < 6.0:                              # wrapped past midnight UTC
            dt += timedelta(days=1)
        out.append(dt)
    return out[0], out[1]


# ── DST transitions (pure; US rule, no tz database) ──────────────────────────
def us_dst_transitions(year: int) -> tuple:
    """(spring_forward_date, fall_back_date) for a year under the post-2007 US
    rule: second Sunday in March, first Sunday in November. Pure calendar
    arithmetic - consistent with this module's no-OS-tz-database principle.
    All stop data here is 2019+, so the pre-2007 rule never applies."""
    from datetime import date, timedelta

    def nth_sunday(month: int, n: int) -> date:
        first = date(year, month, 1)
        days_to_sunday = (6 - first.weekday()) % 7
        return first + timedelta(days=days_to_sunday + 7 * (n - 1))

    return nth_sunday(3, 2), nth_sunday(11, 1)


def dst_season(local_date) -> Optional[str]:
    """'spring' / 'fall' when the date is within DST_WINDOW_DAYS of that year's
    transition, else None. The two windows never overlap (transitions are ~8
    months apart)."""
    spring, fall = us_dst_transitions(local_date.year)
    if abs((local_date - spring).days) <= DST_WINDOW_DAYS:
        return "spring"
    if abs((local_date - fall).days) <= DST_WINDOW_DAYS:
        return "fall"
    return None


# ── Mantel-Haenszel across clock strata (pure) ────────────────────────────────
def mantel_haenszel(strata: list[dict]) -> dict:
    """MH odds ratio + Robins-Breslow-Greenland CI/z over 2x2 strata.

    Each stratum: {day_black, day_nonblack, dark_black, dark_nonblack}.
    Strata with an empty daylight or dark arm contribute nothing and are
    skipped (standard practice). Returns significant=False when the estimate
    is undefined or the data are too thin for a finite variance.
    """
    sum_r = sum_s = 0.0
    sum_pr = sum_ps_qr = sum_qs = 0.0
    for st in strata:
        a, b = st["day_black"], st["day_nonblack"]
        c, d = st["dark_black"], st["dark_nonblack"]
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        r, s = (a * d) / n, (b * c) / n
        p, q = (a + d) / n, (b + c) / n
        sum_r += r
        sum_s += s
        sum_pr += p * r
        sum_ps_qr += p * s + q * r
        sum_qs += q * s
    if sum_r == 0 or sum_s == 0:
        return {"or": None, "ci_low": None, "ci_high": None, "z": None,
                "significant": False}
    or_mh = sum_r / sum_s
    var = (sum_pr / (2 * sum_r * sum_r)
           + sum_ps_qr / (2 * sum_r * sum_s)
           + sum_qs / (2 * sum_s * sum_s))
    se = math.sqrt(var)
    ln_or = math.log(or_mh)
    z = ln_or / se if se else 0.0
    return {
        "or": or_mh,
        "ci_low": math.exp(ln_or - 1.96 * se),
        "ci_high": math.exp(ln_or + 1.96 * se),
        "z": round(z, 2),
        "significant": abs(z) >= 1.96,
    }


def _fmt_minutes(m: int) -> str:
    h, mm = divmod(m, 60)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{mm:02d} {suffix}"


def veil_from_rows(rows: list[dict]) -> dict:
    """Pure computation from per-stop rows:
    {local_date, local_minute, utc_dt, offset_minutes, is_black, ticket_only,
    district}.

    Derives per-date sun events, the pooled inter-twilight window, light/dark
    classification, and TWO Mantel-Haenszel estimates per subset:
      * mh                  strata = 30-minute clock bins (the classic test)
      * mh_time_district    strata = clock bin x district
    The second controls for WHERE policing happens as well as when. If the
    time-only OR departs from 1 but the time-and-district OR does not, the
    citywide lighting pattern is carried by deployment geography rather than
    by lighting itself. Rows with no mappable district are excluded from the
    district-stratified estimate only, and counted.

    Each subset also carries a ``dst`` block - the seasonal-robustness check -
    computed on the rows within DST_WINDOW_DAYS of a daylight-saving
    transition, with BOTH estimates re-derived on strata of clock bin x
    transition season (and x district), so season is held nearly constant and
    spring rows are never pooled against fall rows within a stratum.
    """
    # Per-date solar events, in LOCAL minutes via each date's own UTC offset;
    # the DST-transition season tag rides along (also purely date-derived).
    sun: dict = {}
    for r in rows:
        d = r["local_date"]
        if d in sun:
            continue
        sunset_utc, dusk_utc = sun_events_utc(d)
        off = r["offset_minutes"]
        sun[d] = {
            "sunset_utc": sunset_utc,
            "dusk_utc": dusk_utc,
            "sunset_local_min": (sunset_utc.hour * 60 + sunset_utc.minute + off) % 1440,
            "dusk_local_min": (dusk_utc.hour * 60 + dusk_utc.minute + off) % 1440,
            "season": dst_season(d),
        }
    if not sun:
        raise RuntimeError("no stop rows to test - is the stops table loaded?")

    win_start = min(s["sunset_local_min"] for s in sun.values())
    win_end = max(s["dusk_local_min"] for s in sun.values())

    def new_bucket():
        n_bins = (win_end - win_start + STRATUM_MINUTES - 1) // STRATUM_MINUTES
        return {
            "strata": [
                {"bin_start": win_start + i * STRATUM_MINUTES,
                 "day_black": 0, "day_nonblack": 0,
                 "dark_black": 0, "dark_nonblack": 0}
                for i in range(n_bins)
            ],
            # (bin_idx, district) -> 2x2, built lazily; unmapped districts out.
            "strata_district": defaultdict(
                lambda: {"day_black": 0, "day_nonblack": 0,
                         "dark_black": 0, "dark_nonblack": 0}),
            "district_unmapped": 0,
            "ambiguous": 0, "daylight": [0, 0], "dark": [0, 0],  # [black, total]
            # Transition-window (dst) mirror: season-aware strata + counts.
            # (bin_idx, season) and (bin_idx, season, district) -> 2x2.
            "dst_strata": defaultdict(
                lambda: {"day_black": 0, "day_nonblack": 0,
                         "dark_black": 0, "dark_nonblack": 0}),
            "dst_strata_district": defaultdict(
                lambda: {"day_black": 0, "day_nonblack": 0,
                         "dark_black": 0, "dark_nonblack": 0}),
            "dst_district_unmapped": 0,
            "dst_ambiguous": 0,
            "dst_daylight": [0, 0], "dst_dark": [0, 0],          # [black, total]
        }

    buckets = {"all": new_bucket(), "ticket_only": new_bucket()}
    outside = 0
    for r in rows:
        lm = r["local_minute"]
        if not (win_start <= lm < win_end):
            outside += 1
            continue
        s = sun[r["local_date"]]
        if r["utc_dt"] < s["sunset_utc"]:
            arm = "day"
        elif r["utc_dt"] >= s["dusk_utc"]:
            arm = "dark"
        else:
            arm = None                            # ambiguous twilight
        season = s["season"]
        targets = ["all"] + (["ticket_only"] if r["ticket_only"] else [])
        for t in targets:
            b = buckets[t]
            if arm is None:
                b["ambiguous"] += 1
                if season is not None:
                    b["dst_ambiguous"] += 1
                continue
            idx = (lm - win_start) // STRATUM_MINUTES
            key = f"{arm}_{'black' if r['is_black'] else 'nonblack'}"
            b["strata"][idx][key] += 1
            if r.get("district") is not None:
                b["strata_district"][(idx, r["district"])][key] += 1
            else:
                b["district_unmapped"] += 1
            tab = b["daylight" if arm == "day" else "dark"]
            tab[1] += 1
            if r["is_black"]:
                tab[0] += 1
            if season is not None:                # transition-window mirror
                b["dst_strata"][(idx, season)][key] += 1
                if r.get("district") is not None:
                    b["dst_strata_district"][(idx, season, r["district"])][key] += 1
                else:
                    b["dst_district_unmapped"] += 1
                dtab = b["dst_daylight" if arm == "day" else "dst_dark"]
                dtab[1] += 1
                if r["is_black"]:
                    dtab[0] += 1

    def _arm(tab):
        black, total = tab
        return {"black": black, "total": total,
                "black_share": black / total if total else None}

    def summarize(b):
        return {
            "daylight": _arm(b["daylight"]),
            "dark": _arm(b["dark"]),
            "ambiguous_excluded": b["ambiguous"],
            "mh": mantel_haenszel(b["strata"]),
            "mh_time_district": mantel_haenszel(list(b["strata_district"].values())),
            "district_unmapped_excluded": b["district_unmapped"],
            "strata": b["strata"],
            # Seasonal-robustness check: same estimator, rows within
            # DST_WINDOW_DAYS of a transition, strata refined by season.
            "dst": {
                "days": DST_WINDOW_DAYS,
                "daylight": _arm(b["dst_daylight"]),
                "dark": _arm(b["dst_dark"]),
                "ambiguous_excluded": b["dst_ambiguous"],
                "mh": mantel_haenszel(list(b["dst_strata"].values())),
                "mh_time_district": mantel_haenszel(
                    list(b["dst_strata_district"].values())),
                "district_unmapped_excluded": b["dst_district_unmapped"],
            },
        }

    return {
        "window": {
            "start_minute": win_start, "end_minute": win_end,
            "label": f"{_fmt_minutes(win_start)} to {_fmt_minutes(win_end)}",
        },
        "all": summarize(buckets["all"]),
        "ticket_only": summarize(buckets["ticket_only"]),
        "outside_window_excluded": outside,
        "definition": VEIL_DEFINITION,
    }


# ── DB access ─────────────────────────────────────────────────────────────────
# Evening stops only (a superset of any possible inter-twilight window in DC),
# known race, timed, from the latest finished batch of BOTH layers. The local
# timestamp comes from Postgres; the UTC offset is derived per row so no OS
# timezone database is needed.
_VEIL_ROWS_SQL = """
SELECT stop_datetime AT TIME ZONE 'UTC' AS utc_dt,
       (stop_datetime AT TIME ZONE 'America/New_York')::date AS local_date,
       EXTRACT(HOUR FROM stop_datetime AT TIME ZONE 'America/New_York')::int * 60
         + EXTRACT(MINUTE FROM stop_datetime AT TIME ZONE 'America/New_York')::int
         AS local_minute,
       ROUND(EXTRACT(EPOCH FROM ((stop_datetime AT TIME ZONE 'America/New_York')
         - (stop_datetime AT TIME ZONE 'UTC'))) / 60)::int AS offset_minutes,
       (ethnicity = 'Black') AS is_black,
       (stop_type = 'Ticket Only') AS ticket_only,
       district
FROM stops
WHERE ingest_batch = ANY($1::bigint[])
  AND stop_datetime IS NOT NULL
  AND ethnicity IS NOT NULL AND ethnicity <> 'Unknown'
  AND EXTRACT(HOUR FROM stop_datetime AT TIME ZONE 'America/New_York') BETWEEN 15 AND 22
"""


async def veil_of_darkness(conn, *, layers: tuple = (45, 41)) -> dict:
    """The veil-of-darkness test over the latest finished batch of each layer."""
    batches = {}
    for layer in layers:
        b = await resolve_current_batch(conn, layer)
        if b is not None:
            batches[layer] = b
    if not batches:
        raise RuntimeError("no finished stop-ingest batch - run scripts/ingest_stops.py first.")
    rows = await conn.fetch(_VEIL_ROWS_SQL, list(batches.values()))
    result = veil_from_rows([dict(r) for r in rows])
    result["meta"] = {
        "grain": "stops of persons",
        "layers": batches,
        "definition": VEIL_DEFINITION,
    }
    return result
