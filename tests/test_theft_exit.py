"""Pre-registration tests for the district theft-exit analysis.

These tests pin the finding language and branch-selection rules BEFORE the
estimator exists (the module deliberately contains none). Every branch the
estimate could land in is asserted here against synthetic results, so the
wording on the page can never be tuned to the number it describes.
"""
from database.theft_exit import (
    ANCHOR_KEY,
    ANCHOR_PRE_THRIVE_MONTHS,
    APRIORI_EXCLUSION,
    CELLS,
    CONFOUND_SENTENCES,
    CONTROL_EXCLUSION,
    DID_NULL_SCOPE,
    ENFORCEMENT_EXHIBIT_TEXT,
    EXCLUSION_3D_TEXT,
    NONALIGNMENT_SENTENCE,
    OUTCOME_OFFENSES,
    PRETREND_INFERENCE_TEXT,
    THRIVE_OPENS_BEFORE_FRACTION,
    THRIVE_SPANS,
    THRIVE_TIMING_NOTE,
    active_cells,
    cell_finding_text,
    composition_sentence,
    did_finding_text,
    did_outcome,
    did_plain_answer,
    pretrend_sentence,
    sensitivity_outcome,
    sensitivity_sentence,
    thrive_timing_sentence,
)


# ── Synthetic result builders ─────────────────────────────────────────────────
def _cell(key, district, label, *, beta=-0.02, p_perm=0.5, p_adj=0.6,
          discovery=False, pre_ok=True, pre_p=0.55, below_floor=False,
          doc=15.1, base=80.0, mde=11.0):
    return {
        "key": key, "district": district, "label": label,
        "beta": beta, "se_nw": 0.05,
        "p_perm": p_perm, "p_adjusted": p_adj, "discovery": discovery,
        "pretrend": {"joint_p": pre_p, "ok": pre_ok},
        "bridge": {"doc_per_month": doc, "base_per_month": base,
                   "expected_dip_pct": doc / (doc + base) * 100,
                   "mde_pct": mde, "below_floor": below_floor},
        "docs": next(c for c in CELLS if c["key"] == key)["docs"],
    }


def _result(*, pooled_disc, anchor_disc=False, anchor_pre_ok=True,
            sens_disc=False, cell2_disc=False, cell5_disc=False):
    return {
        "pooled": {"beta": -0.18 if pooled_disc else -0.01,
                   "p_perm": 0.01 if pooled_disc else 0.44,
                   "n_placebo": 149, "discovery": pooled_disc,
                   "weights": {"7D-2024": 0.62, "2D-2025": 0.22,
                               "5D-2025": 0.16}},
        "cells": [
            _cell("7D-2024", "7", "Seventh District, January-October 2024",
                  beta=-0.22 if anchor_disc else -0.03,
                  p_perm=0.008 if anchor_disc else 0.38,
                  p_adj=0.024 if anchor_disc else 0.57,
                  discovery=anchor_disc, pre_ok=anchor_pre_ok,
                  pre_p=0.61 if anchor_pre_ok else 0.03),
            _cell("2D-2025", "2", "Second District, January-August 2025",
                  discovery=cell2_disc, below_floor=True,
                  doc=15.4, base=395.0, mde=9.0),
            _cell("5D-2025", "5",
                  "Fifth District (Rosedale THRIVE), February-August 2025",
                  discovery=cell5_disc, below_floor=True,
                  doc=7.4, base=240.0, mde=12.0),
        ],
        "sensitivity_3d": {"beta": -0.15, "p_perm": 0.03 if sens_disc
                           else 0.21, "discovery": sens_disc},
        "backfill": {"recovered_rows": 0},
    }


# ── Branch selection ──────────────────────────────────────────────────────────
def test_consistent_branch_requires_pooled_and_anchor():
    r = _result(pooled_disc=True, anchor_disc=True)
    assert did_outcome(r) == "consistent"


def test_null_branch_requires_nothing_firing():
    r = _result(pooled_disc=False)
    assert did_outcome(r) == "null"


def test_pooled_without_anchor_is_mixed():
    r = _result(pooled_disc=True, anchor_disc=False)
    assert did_outcome(r) == "mixed"


def test_anchor_without_pooled_is_mixed():
    r = _result(pooled_disc=False, anchor_disc=True)
    assert did_outcome(r) == "mixed"


def test_below_floor_cell_firing_alone_is_mixed_not_consistent():
    r = _result(pooled_disc=False, cell5_disc=True)
    assert did_outcome(r) == "mixed"
    assert "cannot detect the documented scale is exactly the pattern " \
           "noise produces" in did_finding_text(r)


def test_anchor_pretrend_failure_kills_the_design_regardless():
    # Even a firing pooled estimate cannot rescue the design if the only
    # above-floor cell was not parallel beforehand.
    r = _result(pooled_disc=True, anchor_disc=True, anchor_pre_ok=False)
    assert did_outcome(r) == "design_fails"
    text = did_finding_text(r)
    assert "cannot answer the question" in text
    assert "artifact" in text


def test_pretrend_failed_cells_are_not_active():
    r = _result(pooled_disc=True, anchor_disc=True, anchor_pre_ok=False)
    assert [c["key"] for c in active_cells(r)] == ["2D-2025", "5D-2025"]


# ── Finding text per branch ───────────────────────────────────────────────────
def test_consistent_text_carries_floor_claim_and_thrive_ambiguity():
    r = _result(pooled_disc=True, anchor_disc=True)
    text = did_finding_text(r)
    assert "not proof of manipulation" in text
    assert "floor on the documented pattern" in text
    assert "by no means a complete record" in text          # p. 526 caveat
    assert "THRIVE" in text                                  # competing reading
    assert "inseparable" in text
    assert "p. 217" in text                                  # anchor page cite
    # No individual is named in finding text.
    for name in ("Rivers", "Donigian", "Haskis", "Pulliam", "Merrill",
                 "Jenkins", "Makal", "Savoy", "Wright"):
        assert name not in text


def test_null_text_is_about_the_limits_of_public_data():
    r = _result(pooled_disc=False)
    text = did_finding_text(r)
    assert DID_NULL_SCOPE in text
    assert "oversight instrument" in text
    assert "internal audit access" in text


def test_mixed_text_headlines_neither_side():
    fired_pooled = did_finding_text(_result(pooled_disc=True))
    fired_anchor = did_finding_text(_result(pooled_disc=False,
                                            anchor_disc=True))
    for text in (fired_pooled, fired_anchor):
        assert text.startswith("Mixed, and reported as mixed.")
        assert "equal weight" in text
        assert "this page does not make it" in text


def test_plain_answers_cover_all_branches():
    assert "Yes, with the limits stated" in did_plain_answer(
        _result(pooled_disc=True, anchor_disc=True))
    assert did_plain_answer(_result(pooled_disc=False)).startswith("No. ")
    assert "Mixed" in did_plain_answer(_result(pooled_disc=True))
    assert "No answer is possible" in did_plain_answer(
        _result(pooled_disc=True, anchor_disc=True, anchor_pre_ok=False))


# ── Per-cell text ─────────────────────────────────────────────────────────────
def test_below_floor_cells_say_they_cannot_detect():
    r = _result(pooled_disc=False)
    for key in ("2D-2025", "5D-2025"):
        cell = next(c for c in r["cells"] if c["key"] == key)
        text = cell_finding_text(cell)
        assert "below this cell's detection floor" in text
        assert "uninformative by construction" in text
        assert "not as evidence in either direction" in text


def test_pretrend_failure_cell_text_refuses_a_finding():
    r = _result(pooled_disc=True, anchor_disc=True, anchor_pre_ok=False)
    cell = next(c for c in r["cells"] if c["key"] == ANCHOR_KEY)
    text = cell_finding_text(cell)
    assert "The design does not hold here" in text
    assert "no finding, in either direction" in text
    assert "excluded from the pooled estimate" in text


def test_anchor_null_is_labeled_informative():
    r = _result(pooled_disc=False)
    cell = next(c for c in r["cells"] if c["key"] == ANCHOR_KEY)
    text = cell_finding_text(cell)
    assert "informative kind of null" in text
    assert "should have been visible here and is not" in text


# ── Pre-trends as first-class output ──────────────────────────────────────────
def test_clean_pretrends_are_stated_without_overclaiming():
    text = pretrend_sentence(_result(pooled_disc=True, anchor_disc=True))
    assert "necessary condition" in text
    assert "absence of visible violation, not proof of parallelism" in text
    assert "judged by eye" in text
    assert "permutation joint p" in text          # same inference as primary
    assert "strongest" not in text                # superlative dropped


def test_pretrend_gate_uses_design_based_inference():
    assert "same inference as the estimate itself" in PRETREND_INFERENCE_TEXT
    assert "unknown calibration" in PRETREND_INFERENCE_TEXT
    assert "placebo" in PRETREND_INFERENCE_TEXT
    assert "rank of the real statistic" in PRETREND_INFERENCE_TEXT


def test_failed_pretrends_are_named_and_excluded():
    text = pretrend_sentence(_result(pooled_disc=True, anchor_disc=True,
                                     anchor_pre_ok=False))
    assert "Seventh District, January-October 2024" in text
    assert "excluded from the pooled estimate" in text


# ── THRIVE timing call (pre-registered, both directions) ─────────────────────
def _with_timing(r, *, opens_before):
    anchor = next(c for c in r["cells"] if c["key"] == ANCHOR_KEY)
    anchor["thrive_timing"] = {"pre_thrive_mean": -0.15 if opens_before
                               else -0.02, "window_mean": -0.22,
                               "opens_before": opens_before}
    return r


def test_deficit_opening_before_thrive_weakens_initiative_reading():
    r = _with_timing(_result(pooled_disc=True, anchor_disc=True),
                     opens_before=True)
    text = thrive_timing_sentence(r)
    assert "opens before the THRIVE boundary" in text
    assert "May 2, 2024" in text
    assert "weakens the initiative reading" in text
    assert "does not eliminate it" in text        # no overclaim


def test_deficit_opening_after_thrive_keeps_both_readings():
    r = _with_timing(_result(pooled_disc=True, anchor_disc=True),
                     opens_before=False)
    text = thrive_timing_sentence(r)
    assert "at or after the THRIVE boundary" in text
    assert "does not separate" in text
    assert "carries both" in text


def test_no_anchor_deficit_means_no_timing_claim():
    text = thrive_timing_sentence(_result(pooled_disc=False))
    assert "nothing to separate" in text
    assert "no timing claim" in text


def test_timing_rule_constants_are_fixed():
    assert THRIVE_OPENS_BEFORE_FRACTION == 0.5
    assert [str(m) for m in ANCHOR_PRE_THRIVE_MONTHS] == [
        "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]


# ── Composition and sensitivity ───────────────────────────────────────────────
def test_composition_sentence_names_the_real_weights():
    text = composition_sentence(_result(pooled_disc=True, anchor_disc=True))
    assert "mostly the Seventh District 2024 cell (62%)" in text
    assert "22% (2D-2025)" in text and "16% (5D-2025)" in text
    assert "not three independent replications" in text


def test_sensitivity_both_ways_is_free_robustness():
    r = _result(pooled_disc=True, anchor_disc=True, sens_disc=True)
    assert sensitivity_outcome(r) == "both"
    assert "survives both ways" in sensitivity_sentence(r)


def test_sensitivity_primary_only_weakens_the_finding():
    r = _result(pooled_disc=True, anchor_disc=True, sens_disc=False)
    assert sensitivity_outcome(r) == "primary_only"
    text = sensitivity_sentence(r)
    assert "survives only when the Third District is excluded" in text
    assert "weaker" in text


def test_sensitivity_cannot_upgrade_a_null():
    r = _result(pooled_disc=False, sens_disc=True)
    assert sensitivity_outcome(r) == "na_null"
    assert "cannot upgrade a primary null" in sensitivity_sentence(r)


# ── Pre-registered constants ──────────────────────────────────────────────────
def test_apriori_exclusion_states_scale_and_timing():
    assert "robbery-share" in APRIORI_EXCLUSION
    assert "ADW-share" in APRIORI_EXCLUSION
    assert "roughly 5 and 11 reports" in APRIORI_EXCLUSION
    assert "before the estimator existed" in APRIORI_EXCLUSION


def test_confound_block_carries_every_required_confound():
    joined = " ".join(CONFOUND_SENTENCES)
    assert "14% in 2024" in joined                       # citywide decline
    assert "THRIVE" in joined                            # competing reading
    assert "same district-shaped deficit" in joined
    assert "by no means a complete record" in joined     # dirty controls
    assert "floor-style claim" in joined
    assert "never from an editor's assignment" in joined  # attribution rule
    assert "August 2025 federal surge" in joined         # surge boundary
    assert "TPWR" in joined                              # report-only exhibit
    assert "cannot be reproduced from public data" in joined
    assert "post-2012-realignment" in joined             # boundary churn


def test_thrive_timing_note_separates_what_it_can():
    assert "January 2024" in THRIVE_TIMING_NOTE           # 7D window opens
    assert "May 2, 2024" in THRIVE_TIMING_NOTE            # first 7D THRIVE
    assert "inseparable by timing" in THRIVE_TIMING_NOTE  # 5D coincidence
    assert "No THRIVE area operated in the Second District" \
        in THRIVE_TIMING_NOTE


def test_enforcement_exhibit_text_does_not_overclaim():
    assert "cannot rule the initiative reading out" in ENFORCEMENT_EXHIBIT_TEXT
    assert "not a gap left unexamined" in ENFORCEMENT_EXHIBIT_TEXT


# ── Estimator on synthetic data ──────────────────────────────────────────────
# Deterministic monthly counts: district-specific level, shared seasonality,
# and reproducible pseudo-noise. The estimator sees 2021-01 through 2025-07
# (surge-truncated automatically).
#
# The noise MUST be reproducible across processes. It used to use the builtin
# hash(), which Python salts per-process for str keys unless PYTHONHASHSEED is
# set - and it is not set here or in CI. That made every run feed the estimator
# a different dataset: the suite passed on 2026-08-01 and failed on 2026-08-26
# with no code change in between. crc32 is stable everywhere.
#
# NOISE_SEEDS sweeps several fixed draws on purpose. Pinning a single seed
# would freeze one lucky dataset and hide a pretrend gate that is marginal
# under plausible noise - which is exactly what the flaky failure was hinting
# at. If one of these seeds fails, that is a finding about the estimator, not
# a broken test.
import math
import zlib
from datetime import date as _d

import pytest

NOISE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

from database.theft_exit import (CELLS as _CELLS, DISTRICTS, SURGE_START,
                                 _estimable_window, add_months,
                                 theft_exit_from_counts)

_LEVELS = {"1": 220, "2": 400, "3": 340, "4": 220, "5": 240, "6": 145,
           "7": 80}


def _noise(d, m, scale, seed=0):
    key = "{}|{}|{}|{}|tx".format(seed, d, m.year, m.month).encode()
    h = (zlib.crc32(key) % 1000) / 1000.0 - 0.5
    return 1.0 + scale * h


def _month_list():
    out, m = [], _d(2021, 1, 1)
    while m < SURGE_START:
        out.append(m)
        m = add_months(m, 1)
    return out


def _synth_counts(*, deficit=None, pretrend_slope=None, noise=0.06, seed=0):
    """deficit: {district: (start, end_incl, fraction)} multiplicative dip.
    pretrend_slope: {district: monthly log-slope} divergence from 2022 on."""
    counts = {}
    for d in DISTRICTS:
        counts[d] = {}
        for m in _month_list():
            season = 1.0 + 0.10 * math.sin(2 * math.pi * (m.month - 1) / 12)
            v = _LEVELS[d] * season * _noise(d, m, noise, seed)
            if pretrend_slope and d in pretrend_slope:
                t = (m.year - 2022) * 12 + m.month - 1
                if t > 0:
                    v *= math.exp(pretrend_slope[d] * t)
            if deficit and d in deficit:
                s, e, frac = deficit[d]
                if s <= m <= e:
                    v *= (1.0 - frac)
            counts[d][m] = max(1, int(round(v)))
    return counts


def _cell_window(key):
    c = next(c for c in _CELLS if c["key"] == key)
    return c["start"], c["end_incl"]


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_estimator_flat_archive_is_null_with_clean_pretrends(seed):
    result = theft_exit_from_counts(_synth_counts(seed=seed))
    assert did_outcome(result) == "null"
    for c in result["cells"]:
        assert c["pretrend"]["ok"], c["key"]
        assert not c["discovery"]
    assert not result["pooled"]["discovery"]
    assert not result["sensitivity_3d"]["discovery"]


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_estimator_detects_injected_7d_deficit(seed):
    s7, e7 = _cell_window("7D-2024")
    result = theft_exit_from_counts(
        _synth_counts(deficit={"7": (s7, e7, 0.25)}, seed=seed))
    anchor = next(c for c in result["cells"] if c["key"] == ANCHOR_KEY)
    assert anchor["pretrend"]["ok"]
    assert anchor["beta"] < -0.15
    assert anchor["p_perm"] <= 0.05
    assert result["pooled"]["beta"] < 0
    assert result["pooled"]["discovery"]
    assert did_outcome(result) in ("consistent", "mixed")
    # weights are reported and sum to 1 over the three stacks
    w = result["pooled"]["weights"]
    assert set(w) == {"7D-2024", "2D-2025", "5D-2025"}
    assert abs(sum(w.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_estimator_pretrend_divergence_kills_the_anchor(seed):
    result = theft_exit_from_counts(
        _synth_counts(pretrend_slope={"7": -0.02}, seed=seed))
    anchor = next(c for c in result["cells"] if c["key"] == ANCHOR_KEY)
    assert not anchor["pretrend"]["ok"]
    assert did_outcome(result) == "design_fails"


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_estimator_timing_call_fires_for_full_window_deficit(seed):
    # A deficit spanning the whole 7D window includes Jan-Apr 2024, so the
    # pre-registered rule must call it "opens before THRIVE".
    s7, e7 = _cell_window("7D-2024")
    result = theft_exit_from_counts(
        _synth_counts(deficit={"7": (s7, e7, 0.25)}, seed=seed))
    anchor = next(c for c in result["cells"] if c["key"] == ANCHOR_KEY)
    if anchor["discovery"]:
        assert anchor["thrive_timing"]["opens_before"]
        assert "opens before the THRIVE boundary" in \
            thrive_timing_sentence(result)


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_estimator_bridges_report_live_bases(seed):
    result = theft_exit_from_counts(_synth_counts(seed=seed))
    for c in result["cells"]:
        b = c["bridge"]
        assert b["base_per_month"] > 0
        assert 0 < b["expected_dip_pct"] < 100
        assert b["mde_pct"] > 0
        # Ensembles are real, not a handful: the anchor stack has the long
        # span (60+ placebos); the surge-truncated 2025 stacks are smaller.
        floor = 50 if c["key"] == ANCHOR_KEY else 15
        assert c["n_placebo"] >= floor, (c["key"], c["n_placebo"])


def test_design_constants_are_the_approved_ones():
    assert OUTCOME_OFFENSES == ("THEFT/OTHER", "THEFT F/AUTO")
    assert [c["key"] for c in CELLS] == ["7D-2024", "2D-2025", "5D-2025"]
    # Every cell carries at least one page cite into the page.
    for c in CELLS:
        assert all(p.startswith("p") for _, p in c["docs"])
    assert CONTROL_EXCLUSION["district"] == "3"
    assert str(CONTROL_EXCLUSION["start"]) == "2024-06-01"   # over-wide
    assert EXCLUSION_3D_TEXT.count("p. ") >= 3               # cites 16/71/15
    assert "deliberately" in EXCLUSION_3D_TEXT
    assert THRIVE_SPANS["2"] == ()                            # no 2D THRIVE
    assert "live dispute" in NONALIGNMENT_SENTENCE            # same as citywide
