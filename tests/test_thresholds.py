"""Threshold-test engine + page: pure helpers, model math, stored-fit render.

The MCMC itself runs offline (scripts/fit_threshold_test.py) and is not
exercised in CI; what IS tested is everything the published numbers rest on:
the truncated-Beta search/hit-rate identities, the likelihood, the summary
shapes, the significance gating, and the page render from a stored fit.
Model-math tests skip cleanly where scipy is absent (it is a fit-time
dependency, not a deployment one).
"""
from datetime import datetime

import pytest

from database.thresholds import gap_direction, ppc_quality


# ── Pure interpretation helpers ───────────────────────────────────────────────
def test_gap_direction_gated_on_interval():
    assert gap_direction({"ci_low": 0.01, "ci_high": 0.09}) == {
        "direction": "black_lower", "significant": True}
    assert gap_direction({"ci_low": -0.09, "ci_high": -0.01}) == {
        "direction": "black_higher", "significant": True}
    assert gap_direction({"ci_low": -0.02, "ci_high": 0.05})["significant"] is False
    assert gap_direction({})["direction"] == "not comparable"


def test_ppc_quality_is_predictive_coverage():
    good = {"cells_checked": 14, "search_cells_outside_95pi": 0,
            "hit_cells_outside_95pi": 0,
            "max_abs_search_rate_err": 0.002, "max_abs_hit_rate_err": 0.06}
    assert ppc_quality(good)["ok"] is True          # thin-cell noise tolerated
    bad = dict(good, hit_cells_outside_95pi=1)
    assert ppc_quality(bad)["ok"] is False
    assert ppc_quality({})["ok"] is False


# ── Model math (fit-time deps; skipped where scipy is absent) ─────────────────
def test_truncated_beta_identities_match_numerical_integration():
    scipy = pytest.importorskip("scipy")
    import numpy as np
    from scipy import integrate
    from scipy.stats import beta as beta_dist

    from scripts.threshold_model import search_and_hit_rates

    phi, lam, t = 0.15, 5.0, 0.2
    S, H = search_and_hit_rates(np.array(phi), np.array(lam), np.array(t))
    a, b = phi * lam, (1 - phi) * lam
    S_num, _ = integrate.quad(lambda p: beta_dist.pdf(p, a, b), t, 1)
    H_num, _ = integrate.quad(lambda p: p * beta_dist.pdf(p, a, b), t, 1)
    assert float(S) == pytest.approx(S_num, rel=1e-6)
    assert float(H) == pytest.approx(H_num / S_num, rel=1e-6)


def test_log_posterior_finite_and_vectorized():
    pytest.importorskip("scipy")
    import numpy as np

    from scripts.threshold_model import initial_ensemble, log_posterior, n_params

    D, R = 7, 2
    rng = np.random.default_rng(0)
    data = {"stops": rng.integers(1000, 5000, (D, R)),
            "searches": rng.integers(50, 300, (D, R)),
            "hits": rng.integers(5, 40, (D, R))}
    theta = initial_ensemble(rng, 12, D, R, data)
    lp = log_posterior(theta, data, D, R)
    assert lp.shape == (12,)
    assert np.all(np.isfinite(lp))
    # Single-point call agrees with the vectorized path.
    assert log_posterior(theta[0], data, D, R) == pytest.approx(lp[0])


def test_summarize_fit_shapes_and_gap_sign():
    pytest.importorskip("scipy")
    import numpy as np

    from scripts.threshold_model import initial_ensemble, n_params, summarize_fit

    D, R = 7, 2
    rng = np.random.default_rng(1)
    data = {"stops": np.full((D, R), 1000), "searches": np.full((D, R), 100),
            "hits": np.full((D, R), 20)}
    chain = initial_ensemble(rng, 200, D, R, data)
    out = summarize_fit(chain, data, D, R, list(range(1, 8)),
                        ["black", "non_black_known_race"])
    assert set(out["citywide_thresholds"]) == {"black", "non_black_known_race"}
    assert len(out["cells"]) == 14
    assert {"mean", "ci_low", "ci_high", "p_black_lower"} <= set(out["gap"])
    assert out["ppc"]["max_abs_search_rate_err"] >= 0
    assert out["ppc"]["cells_checked"] == 14
    assert "search_cells_outside_95pi" in out["ppc"]


def test_build_cells_person_level_and_exclusions():
    pytest.importorskip("scipy")
    from scripts.fit_threshold_test import build_cells

    rows = [
        # District 2, Black: 10 stops, one pat-down with a recorded hit.
        {"district": 2, "ethnicity": "Black", "search_consent": False,
         "search_warrant": False, "pat_down": True, "search_prob_cause": False,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": "Gun ;", "n": 1},
        {"district": 2, "ethnicity": "Black", "search_consent": False,
         "search_warrant": False, "pat_down": False, "search_prob_cause": False,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": None, "n": 9},
        # Pat-down with NO recorded outcome: excluded from searches, counted.
        {"district": 2, "ethnicity": "White", "search_consent": False,
         "search_warrant": False, "pat_down": True, "search_prob_cause": False,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": "NA", "n": 2},
        # Probable-cause-only search: excluded, counted.
        {"district": 3, "ethnicity": "White", "search_consent": False,
         "search_warrant": False, "pat_down": False, "search_prob_cause": True,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": None, "n": 3},
        # Unknown race and unmapped district: excluded, counted.
        {"district": 2, "ethnicity": "Unknown", "search_consent": False,
         "search_warrant": False, "pat_down": False, "search_prob_cause": False,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": None, "n": 4},
        {"district": None, "ethnicity": "Black", "search_consent": False,
         "search_warrant": False, "pat_down": False, "search_prob_cause": False,
         "seized_consent_raw": None, "seized_warrant_raw": None,
         "seized_pat_down_raw": None, "n": 5},
    ]
    data, excl = build_cells(rows)
    assert data["stops"][1, 0] == 10          # D2 Black
    assert data["searches"][1, 0] == 1 and data["hits"][1, 0] == 1
    assert data["stops"][1, 1] == 2           # D2 White counted as stops
    assert data["searches"][1, 1] == 0        # unrecorded outcome -> not a model search
    assert excl == {"unknown_race": 4, "unmapped_district": 5,
                    "prob_cause_only_searches": 3, "outcome_unrecorded_searches": 2}


# ── Page render from a stored fit ─────────────────────────────────────────────
def _fit(gap_low=0.02, gap_high=0.08, s_out=0, h_out=0):
    return {
        "fit_id": 1, "created_at": datetime(2026, 7, 29),
        "model_version": "beta-signal-hier-v1", "source_layer": 41, "stop_batch": 5,
        "citywide_thresholds": {
            "black": {"mean": 0.16, "ci_low": 0.14, "ci_high": 0.18},
            "non_black_known_race": {"mean": 0.21, "ci_low": 0.18, "ci_high": 0.24},
        },
        "gap": {"mean": (gap_low + gap_high) / 2, "ci_low": gap_low,
                "ci_high": gap_high, "p_black_lower": 0.99},
        "cells": [
            {"district": d, "race": r, "stops": 10000, "searches": 400, "hits": 60,
             "threshold": {"mean": 0.17, "ci_low": 0.15, "ci_high": 0.19},
             "search_rate_obs": 0.04, "hit_rate_obs": 0.15}
            for d in range(1, 8) for r in ("black", "non_black_known_race")
        ],
        "ppc": {"cells_checked": 14, "search_cells_outside_95pi": s_out,
                "hit_cells_outside_95pi": h_out,
                "max_abs_search_rate_err": 0.004, "max_abs_hit_rate_err": 0.012},
        "diagnostics": {"sampler": "emcee affine-invariant ensemble",
                        "posterior_samples": 4000, "mean_acceptance": 0.24,
                        "converged": True},
        "exclusions": {"unknown_race": 15000, "unmapped_district": 3000,
                       "prob_cause_only_searches": 2000,
                       "outcome_unrecorded_searches": 900},
    }
