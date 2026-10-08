"""Tests for the classification-mix changepoint engine (pure math).

The stakes: this page may face adversarial review from quant-literate readers
and from MPD's own data office. The load-bearing properties are that the
machinery does NOT manufacture findings: a flat series reports no break, an
autocorrelated flat series that naive OLS would flag is correctly not flagged
under Newey-West, and nine null series survive the corrected family test with
zero discoveries. Detection power is tested against a synthetic series with a
known injected break. All randomness is seeded; every test is deterministic.
"""
import math
import random
from datetime import date

from database.classification_mix import (
    PAIRS,
    VICTIM_OFFENSES,
    add_months,
    benjamini_hochberg,
    binary_segmentation,
    classification_mix_from_matrix,
    deseasonalize,
    its_level_shift,
    month_range,
    newey_west_lags,
    pair_share,
    share_matrix,
)

MONTHS = month_range(date(2008, 1, 1), date(2026, 7, 1))   # 222 months


def _seasonal(m: date) -> float:
    return 0.01 * math.sin(2 * math.pi * (m.month - 1) / 12.0)


# ── Share computation ─────────────────────────────────────────────────────────
def test_share_matrix_zero_fills_and_shares_sum_to_one():
    start, end = date(2024, 1, 1), date(2024, 4, 1)
    rows = [
        {"month": date(2024, 1, 1), "offense": "ROBBERY", "n": 25},
        {"month": date(2024, 1, 1), "offense": "THEFT/OTHER", "n": 75},
        # Feb: nothing at all (a gap month must zero-fill, not vanish).
        {"month": date(2024, 3, 1), "offense": "HOMICIDE", "n": 10},
    ]
    m = share_matrix(rows, start, end)
    assert m["months"] == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    assert m["totals"] == [100, 0, 10]
    assert m["counts"]["ROBBERY"] == [25, 0, 0]
    assert m["shares"]["ROBBERY"][0] == 0.25
    assert m["shares"]["THEFT/OTHER"][0] == 0.75
    # A zero-incident month yields None shares, never a fabricated number.
    assert all(m["shares"][off][1] is None for off in VICTIM_OFFENSES)
    assert sum(m["shares"][off][0] for off in VICTIM_OFFENSES) == 1.0
    assert m["shares"]["HOMICIDE"][2] == 1.0


def test_share_matrix_ignores_unknown_categories_and_out_of_window_rows():
    start, end = date(2024, 1, 1), date(2024, 2, 1)
    rows = [
        {"month": date(2024, 1, 1), "offense": "ROBBERY", "n": 5},
        {"month": date(2024, 1, 1), "offense": "SIMPLE ASSAULT", "n": 99},
        {"month": date(2023, 12, 1), "offense": "ROBBERY", "n": 99},
    ]
    m = share_matrix(rows, start, end)
    assert m["totals"] == [5]
    assert m["shares"]["ROBBERY"][0] == 1.0


def test_pair_share_is_immune_to_third_category_surges():
    counts = {"THEFT F/AUTO": [30, 30], "THEFT/OTHER": [70, 70],
              "MOTOR VEHICLE THEFT": [10, 500]}
    y = pair_share(counts, "THEFT F/AUTO", ("THEFT/OTHER",))
    # The MVT surge in month 2 cannot move the within-pair share.
    assert y == [0.3, 0.3]


# ── Changepoint detection ─────────────────────────────────────────────────────
def _seasonal_noise_series(seed: int, shift_at: int | None = None,
                           shift: float = 0.05) -> list[float]:
    rng = random.Random(seed)
    out = []
    for i, m in enumerate(MONTHS):
        v = 0.30 + _seasonal(m) + rng.gauss(0, 0.008)
        if shift_at is not None and i >= shift_at:
            v += shift
        out.append(v)
    return out


def test_binseg_detects_a_known_break():
    y = _seasonal_noise_series(seed=7, shift_at=120)
    breaks = binary_segmentation(deseasonalize(y, MONTHS))
    assert breaks == [120]


def test_binseg_reports_no_break_on_a_flat_series():
    y = _seasonal_noise_series(seed=7, shift_at=None)
    assert binary_segmentation(deseasonalize(y, MONTHS)) == []


def test_binseg_respects_minimum_segment_length():
    # A break 6 months from the end cannot be reported with min_seg=12.
    y = _seasonal_noise_series(seed=7, shift_at=len(MONTHS) - 6, shift=0.2)
    breaks = binary_segmentation(deseasonalize(y, MONTHS))
    assert all(12 <= b <= len(MONTHS) - 12 for b in breaks)


# ── Interrupted time series ───────────────────────────────────────────────────
def test_its_recovers_an_injected_level_shift():
    y = _seasonal_noise_series(seed=7, shift_at=120, shift=0.05)
    r = its_level_shift(y, MONTHS, break_month=MONTHS[120])
    assert r["ci_low"] <= 0.05 <= r["ci_high"]
    assert r["p_nw"] < 0.001


def test_its_flat_series_is_not_significant():
    y = _seasonal_noise_series(seed=7, shift_at=None)
    r = its_level_shift(y, MONTHS)
    assert r["p_nw"] > 0.05


def test_newey_west_rescues_the_autocorrelated_false_positive():
    """The one failure mode this page cannot afford: an AR(1) series with NO
    true break that naive OLS flags as a significant 2013 shift. Newey-West
    must not flag it. (seed=7 chosen deterministically: p_ols=.006, p_nw=.11)"""
    rng = random.Random(7)
    phi, prev, y = 0.9, 0.0, []
    for m in MONTHS:
        prev = phi * prev + rng.gauss(0, 0.004)
        y.append(0.30 + _seasonal(m) + prev)
    r = its_level_shift(y, MONTHS)
    assert r["p_ols"] < 0.05          # naive OLS would headline this
    assert r["p_nw"] >= 0.10          # the corrected test correctly does not


def test_newey_west_lag_rule():
    assert newey_west_lags(222) == 4
    assert newey_west_lags(100) == 4
    assert newey_west_lags(50) == 3


# ── Multiple testing ──────────────────────────────────────────────────────────
def test_benjamini_hochberg_known_vector():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    bh = benjamini_hochberg(p, q=0.05)
    # Classic textbook outcome: only the first two survive at q=.05.
    assert bh["discoveries"] == [True, True] + [False] * 8
    assert bh["family_size"] == 10
    # Adjusted p-values are monotone and >= raw.
    assert all(a >= r for a, r in zip(bh["adjusted"], p))
    assert bh["adjusted"][0] == 0.01


def test_nine_null_series_yield_zero_corrected_discoveries():
    """With 9 series, chance alone gives ~0.45 uncorrected 'significant'
    breaks at the 5% level; the corrected family must report none."""
    rng = random.Random(0)
    pvals = []
    for _ in range(9):
        y = [0.2 + _seasonal(m) + rng.gauss(0, 0.01) for m in MONTHS]
        pvals.append(its_level_shift(y, MONTHS)["p_nw"])
    bh = benjamini_hochberg(pvals)
    assert not any(bh["discoveries"])


def test_a_real_shift_survives_the_correction_among_nulls():
    rng = random.Random(0)
    pvals = []
    for i in range(9):
        shift_at = 60 if i == 0 else None
        y = []
        for t, m in enumerate(MONTHS):
            v = 0.2 + _seasonal(m) + rng.gauss(0, 0.01)
            if shift_at is not None and t >= shift_at:
                v += 0.06
            y.append(v)
        pvals.append(its_level_shift(y, MONTHS)["p_nw"])
    bh = benjamini_hochberg(pvals)
    assert bh["discoveries"][0] is True
    assert sum(bh["discoveries"]) == 1


# ── End-to-end assembly ───────────────────────────────────────────────────────
def _synthetic_matrix(shift_offense: str | None = None, shift_at: int = 60,
                      seed: int = 11) -> dict:
    """A full synthetic archive: stable counts per category with seasonal and
    noise structure, optionally one category stealing volume from
    THEFT/OTHER at a given month (a substitution-shaped break)."""
    rng = random.Random(seed)
    base = {"THEFT/OTHER": 1200, "THEFT F/AUTO": 800, "MOTOR VEHICLE THEFT": 300,
            "ROBBERY": 250, "BURGLARY": 200, "ASSAULT W/DANGEROUS WEAPON": 160,
            "SEX ABUSE": 25, "HOMICIDE": 15, "ARSON": 5}
    rows = []
    for i, m in enumerate(MONTHS):
        season = 1.0 + 0.1 * math.sin(2 * math.pi * (m.month - 1) / 12.0)
        for off, b in base.items():
            n = b * season * rng.uniform(0.95, 1.05)
            if shift_offense and i >= shift_at:
                if off == shift_offense:
                    n *= 1.35
                elif off == "THEFT/OTHER":
                    n -= 0.35 * base[shift_offense] * season
            rows.append({"month": m, "offense": off, "n": max(1, round(n))})
    return share_matrix(rows, MONTHS[0], add_months(MONTHS[-1], 1))


def test_from_matrix_flat_archive_reports_no_discoveries_and_no_2013_breaks():
    result = classification_mix_from_matrix(_synthetic_matrix())
    assert result["meta"]["family_size"] == len(VICTIM_OFFENSES) + len(PAIRS)
    for s in result["series"]:
        assert s["its"]["discovery"] is False
        assert s["breaks"] == []
    # The specificity check runs, and on a flat archive placebos are quiet too.
    placebos = result["meta"]["placebos"]
    assert [p["month"] for p in placebos] == [
        date(2011, 1, 1), date(2012, 1, 1), date(2014, 1, 1), date(2015, 1, 1)]
    assert all(p["discoveries"] == 0 for p in placebos)


def test_from_matrix_substitution_break_is_found_and_dated():
    result = classification_mix_from_matrix(
        _synthetic_matrix(shift_offense="THEFT F/AUTO", shift_at=60))
    by_key = {s["key"]: s for s in result["series"]}
    gained = by_key["THEFT F/AUTO"]
    assert gained["its"]["discovery"] is True
    assert gained["its"]["shift"] > 0
    # binseg dates the break within a couple of months of the injection.
    assert any(abs((b - MONTHS[60]).days) < 93 for b in gained["breaks"])
    # The within-pair substitution series sees it too.
    assert by_key["theft_split"]["its"]["discovery"] is True
    # An untouched category far from the substitution does not light up.
    assert by_key["HOMICIDE"]["its"]["discovery"] is False


def test_from_matrix_carries_the_confound_metadata():
    result = classification_mix_from_matrix(_synthetic_matrix())
    meta = result["meta"]
    assert meta["break_month"] == date(2013, 1, 1)
    assert [e["key"] for e in meta["events"]] == [
        "sru", "mark43", "smith", "exposure", "training"]
    for e in meta["events"]:
        # Reference lines are labeled with what happened, not just a year.
        assert len(e["label"]) > 40 and e["short"]
    assert meta["covid_band"][0] == date(2020, 3, 1)
    # The federal surge band runs from Aug 2025 to the end of the data and is
    # labeled as an attribution boundary, like COVID.
    assert meta["surge_band"][0] == date(2025, 8, 1)
    assert "cannot" in meta["surge_label"] or "nothing after" in meta["surge_label"]
    # Feb 2026 through Jun 2026 inclusive = 5 complete post-training months.
    assert meta["post_training_months"] == 5
    assert meta["nw_lags"] == 4


# ── Break attribution buckets ────────────────────────────────────────────────
def test_classify_break_buckets():
    from database.classification_mix import classify_break
    assert classify_break(date(2020, 4, 1)) == "covid"
    # Monthly grain can place a boundary-caused break a month or two early:
    # boundary-adjacent breaks get the boundary's attribution, not a
    # nearest-dashed-line label.
    assert classify_break(date(2020, 2, 1)) == "covid_boundary"
    assert classify_break(date(2020, 1, 1)) == "covid_boundary"
    assert classify_break(date(2019, 12, 1)) == "other"
    assert classify_break(date(2025, 8, 1)) == "surge_era"
    assert classify_break(date(2026, 1, 1)) == "surge_era"
    assert classify_break(date(2025, 7, 1)) == "surge_boundary"
    assert classify_break(date(2025, 6, 1)) == "surge_boundary"
    assert classify_break(date(2025, 5, 1)) == "other"
    assert classify_break(date(2016, 1, 1)) == "other"


# ── The documented manipulation era (Question 2) ─────────────────────────────
def test_one_sided_p_direction_logic():
    from database.classification_mix import one_sided_p
    # Shift in the predicted direction: half the two-sided p.
    assert one_sided_p(-0.02, 0.01, -1) < 0.05
    # Shift OPPOSITE the prediction: p near 1, never significant.
    assert one_sided_p(+0.02, 0.01, -1) > 0.95
    assert one_sided_p(0.0, 0.0, -1) == 1.0


def test_era_window_avoids_covid_and_surge_by_construction():
    from database.classification_mix import ERA_BREAK, ERA_WINDOW_MONTHS
    lo = add_months(ERA_BREAK, -ERA_WINDOW_MONTHS)
    hi = add_months(ERA_BREAK, ERA_WINDOW_MONTHS)
    assert lo == date(2021, 8, 1)      # after the COVID band ends (Jun 2021)
    assert hi == date(2025, 8, 1)      # ends exactly at the surge boundary


def test_era_its_runs_on_the_24_month_window():
    result = classification_mix_from_matrix(_synthetic_matrix())
    s = result["series"][0]
    assert s["era_its"]["n"] == 48
    assert s["era_its"]["nw_lags"] == 3
    assert s["era_its"]["window"] == (date(2021, 8, 1), date(2025, 7, 1))
    assert result["era"]["family_size"] == 12
    # Era placebos: pre-COVID Augusts with the same window shape.
    assert [p["month"] for p in result["era"]["placebos"]] == [
        date(2011, 8, 1), date(2013, 8, 1), date(2015, 8, 1), date(2017, 8, 1)]


def _era_manipulation_matrix(revert: bool, seed: int = 23) -> dict:
    """Synthetic archive with the aggregate signature of the documented
    manipulation: from Aug 2023, robberies logged as theft and ADW downgraded
    out of the feed; optionally reverting after the Dec 2025 exposure."""
    rng = random.Random(seed)
    base = {"THEFT/OTHER": 1200, "THEFT F/AUTO": 800, "MOTOR VEHICLE THEFT": 300,
            "ROBBERY": 250, "BURGLARY": 200, "ASSAULT W/DANGEROUS WEAPON": 160,
            "SEX ABUSE": 25, "HOMICIDE": 15, "ARSON": 5}
    era_start = date(2023, 8, 1)
    expo = date(2025, 12, 1)
    rows = []
    for m in MONTHS:
        season = 1.0 + 0.1 * math.sin(2 * math.pi * (m.month - 1) / 12.0)
        pressured = m >= era_start and not (revert and m >= expo)
        for off, b in base.items():
            n = b * season * rng.uniform(0.97, 1.03)
            if pressured:
                if off == "ROBBERY":
                    n *= 0.70                       # robberies logged as theft
                elif off == "THEFT/OTHER":
                    n += 0.30 * base["ROBBERY"] * season
                elif off == "ASSAULT W/DANGEROUS WEAPON":
                    n *= 0.75                       # downgrades exit the feed
            rows.append({"month": m, "offense": off, "n": max(1, round(n))})
    return share_matrix(rows, MONTHS[0], add_months(MONTHS[-1], 1))


def test_directional_family_detects_the_manipulation_signature():
    result = classification_mix_from_matrix(_era_manipulation_matrix(revert=True))
    d = {e["key"]: e for e in result["era"]["directional"]}
    assert d["robbery_theft"]["discovery"] is True
    # ADW's global share falls (downgrades exit the feed). Its share OF
    # violent barely moves because robbery shrinks alongside it, which is
    # itself a lesson the page states: contrasts are not interchangeable.
    assert d["ASSAULT W/DANGEROUS WEAPON"]["discovery"] is True
    assert d["robbery_theft"]["its"]["shift"] < 0     # direction as predicted
    # Reversion after exposure fires, completing the signature.
    assert d["robbery_theft"]["reversion_discovery"] is True
    assert d["robbery_theft"]["signature"] is True
    assert d["ASSAULT W/DANGEROUS WEAPON"]["signature"] is True


def test_no_reversion_means_no_signature():
    result = classification_mix_from_matrix(_era_manipulation_matrix(revert=False))
    d = {e["key"]: e for e in result["era"]["directional"]}
    assert d["robbery_theft"]["discovery"] is True    # the decline is there
    assert d["robbery_theft"]["reversion_discovery"] is False
    assert d["robbery_theft"]["signature"] is False   # but no reversion leg


def test_flat_archive_era_tests_report_nothing():
    result = classification_mix_from_matrix(_synthetic_matrix())
    for e in result["era"]["directional"]:
        assert e["discovery"] is False
        assert e["signature"] is False


def test_magnitude_bridge_arithmetic():
    result = classification_mix_from_matrix(_synthetic_matrix())
    m = result["era"]["magnitude"]
    # The computed sum from the primary document's per-official audits, and
    # it must equal the sum of the page-cited entries it is built from.
    assert m["documented_reports"] == 922
    assert m["documented_reports"] == sum(n for _, n, _ in m["official_counts"])
    assert all(page.startswith("p. ") for _, _, page in m["official_counts"])
    assert m["era_months"] == 28                      # Aug 2023 .. Nov 2025
    assert m["path_floors"] == {"robbery": 5, "adw": 11, "theft_exit": 406}
    vol = m["monthly_volumes"]
    # Documented robbery path: 5 reports over the era against robbery+theft.
    expect = (5 / 28) / (vol["robbery"] + vol["theft_other"]) * 100
    assert abs(m["doc_robbery_pp"] - expect) < 1e-9
    # Stress ceiling: every edit on the robbery path.
    expect_ceiling = (922 / 28) / (vol["robbery"] + vol["theft_other"]) * 100
    assert abs(m["ceiling_robbery_pp"] - expect_ceiling) < 1e-9
    # The ceiling dwarfs the documented path; both compare to the same MDE.
    assert m["ceiling_robbery_pp"] > 100 * m["doc_robbery_pp"]
    assert abs(m["theft_exit_per_month"] - 406 / 28) < 1e-9
    assert m["theft_exit_feed_pct"] > 0
    assert m["mde_pp"]["robbery_theft"] > 0
    assert isinstance(m["below_detectability"], bool)
