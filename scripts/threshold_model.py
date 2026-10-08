"""Bayesian threshold model (Simoiu, Corbett-Davies & Goel 2017) — pure math.

Fit-time only: imported by scripts/fit_threshold_test.py and the tests, never
by the serving app (scipy/emcee are not deployment dependencies).

MODEL. For stops in cell c = (district d, race r), the officer's assessed
probability p that the person carries contraband is distributed
    p ~ Beta(phi_c * lam_c, (1 - phi_c) * lam_c)
and a search happens iff p exceeds the cell's latent threshold t_c. Then
    search rate  S_c = P(p > t_c)
    hit rate     H_c = E[p | p > t_c]
using the identity E[p; p > t] = phi * SF(t; a+1, b) with a = phi*lam,
b = (1-phi)*lam. Likelihood: searches ~ Binomial(stops, S_c) and hits ~
Binomial(searches, H_c).

IDENTIFICATION comes from the hierarchy (each cell alone gives two statistics
for three parameters): signal parameters share race intercepts and district
offsets; thresholds are drawn per cell from race-level normal priors. This is
the paper's structure with the same qualitative priors, sampled with emcee
(affine-invariant ensemble; no gradients, no compiled dependencies).

IDENTIFICATION NOTE. The original paper fits per-cell thresholds because it
has ~100 departments x 4 races (~400 cells). DC has 7 districts x 2 groups =
14 cells, and per-cell thresholds left the posterior multi-modal (two runs
disagreed qualitatively). Thresholds here therefore carry ADDITIVE race +
district structure - t_cell = logit^-1(t_race + t_district) - which is exactly
the quantity the page reports (the race gap) and what 14 cells can identify.

Parameter vector (D districts, R races; defaults D=7, R=2), packed in order:
    phi0[R], lam0[R],                     race intercepts (logit / log scale)
    phid[D], lamd[D],                     signal district offsets
    t_r[R], t_d[D],                       threshold race + district structure
    log_sig_t, log_sig_phi, log_sig_lam   hierarchical scales
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm

EPS = 1e-9


def n_params(D: int, R: int) -> int:
    return 2 * R + 2 * D + R + D + 3


def unpack(theta: np.ndarray, D: int, R: int) -> dict:
    """theta (..., n_params) -> dict of named parameter arrays (broadcasting
    over leading dimensions, so the same code serves one point or a walker
    ensemble)."""
    i = 0
    def take(k):
        nonlocal i
        out = theta[..., i:i + k]
        i += k
        return out
    return {
        "phi0": take(R), "lam0": take(R),
        "phid": take(D), "lamd": take(D),
        "t_r": take(R), "t_d": take(D),
        "log_sig_t": take(1), "log_sig_phi": take(1), "log_sig_lam": take(1),
    }


def cell_params(p: dict, D: int, R: int) -> tuple:
    """(phi, lam, t) per cell, each shaped (..., D, R)."""
    phi = expit(p["phi0"][..., None, :] + p["phid"][..., :, None])
    lam = np.exp(np.clip(p["lam0"][..., None, :] + p["lamd"][..., :, None], -3, 8))
    t = expit(p["t_r"][..., None, :] + p["t_d"][..., :, None])
    return phi, lam, t


def search_and_hit_rates(phi, lam, t) -> tuple:
    """S = P(p > t), H = E[p | p > t] for Beta(phi*lam, (1-phi)*lam)."""
    phi = np.clip(phi, EPS, 1 - EPS)
    a = phi * lam
    b = (1.0 - phi) * lam
    sf = beta_dist.sf(t, a, b)
    sf_up = beta_dist.sf(t, a + 1.0, b)
    S = np.clip(sf, EPS, 1 - EPS)
    H = np.clip(phi * sf_up / S, EPS, 1 - EPS)
    return S, H


def log_prior(p: dict, D: int, R: int) -> np.ndarray:
    """Weakly-informative priors mirroring the paper's structure. Scale
    parameters are clipped on the log scale so wandering walkers cannot
    overflow exp() during early exploration."""
    log_sig_t = np.clip(p["log_sig_t"][..., 0], -6, 3)
    log_sig_phi = np.clip(p["log_sig_phi"][..., 0], -6, 3)
    log_sig_lam = np.clip(p["log_sig_lam"][..., 0], -6, 3)
    sig_t, sig_phi, sig_lam = np.exp(log_sig_t), np.exp(log_sig_phi), np.exp(log_sig_lam)

    lp = norm.logpdf(p["phi0"], -2.0, 1.5).sum(axis=-1)          # carry ~ 12%
    lp = lp + norm.logpdf(p["lam0"], 1.0, 1.0).sum(axis=-1)
    lp = lp + norm.logpdf(p["t_r"], -2.0, 1.5).sum(axis=-1)      # thresholds ~ 2-50%
    # Half-normal scales via log-parameterization (+ log-Jacobian).
    lp = lp + norm.logpdf(sig_t, 0, 1.0) + np.log(2.0) + log_sig_t
    lp = lp + norm.logpdf(sig_phi, 0, 1.0) + np.log(2.0) + log_sig_phi
    lp = lp + norm.logpdf(sig_lam, 0, 1.0) + np.log(2.0) + log_sig_lam
    # Hierarchical terms.
    lp = lp + norm.logpdf(p["phid"], 0, sig_phi[..., None]).sum(axis=-1)
    lp = lp + norm.logpdf(p["lamd"], 0, sig_lam[..., None]).sum(axis=-1)
    lp = lp + norm.logpdf(p["t_d"], 0, sig_t[..., None]).sum(axis=-1)
    return lp


def log_likelihood(p: dict, data: dict, D: int, R: int) -> np.ndarray:
    """data: stops/searches/hits each (D, R) int arrays. Cells with zero stops
    contribute nothing (masked)."""
    phi, lam, t = cell_params(p, D, R)
    S, H = search_and_hit_rates(phi, lam, t)
    stops, searches, hits = data["stops"], data["searches"], data["hits"]
    mask = stops > 0
    ll_s = np.where(mask, binom.logpmf(searches, stops, S), 0.0)
    ll_h = np.where(mask & (searches > 0), binom.logpmf(hits, searches, H), 0.0)
    return (ll_s + ll_h).sum(axis=(-1, -2))


def log_posterior(theta: np.ndarray, data: dict, D: int, R: int) -> np.ndarray:
    """Vectorized over walkers: theta (nwalkers, n_params) or (n_params,)."""
    p = unpack(np.asarray(theta, dtype=float), D, R)
    lp = log_prior(p, D, R) + log_likelihood(p, data, D, R)
    return np.where(np.isfinite(lp), lp, -1e12)


def initial_ensemble(rng: np.random.Generator, nwalkers: int, D: int, R: int,
                     data: dict) -> np.ndarray:
    """Walkers start near data-informed values: phi near the observed hit
    rates, thresholds near the observed search-rate quantiles."""
    obs_hit = data["hits"].sum() / max(data["searches"].sum(), 1)
    center = np.zeros(n_params(D, R))
    idx = 0
    center[idx:idx + R] = np.log(obs_hit / (1 - obs_hit)); idx += R      # phi0
    center[idx:idx + R] = 1.0; idx += R                                  # lam0
    idx += 2 * D                                                         # signal offsets 0
    center[idx:idx + R] = -2.0; idx += R                                 # t_r
    idx += D                                                             # t_d 0
    center[idx:] = np.log(0.5)                                           # log scales
    return center + 0.1 * rng.standard_normal((nwalkers, center.size))


def summarize_fit(chain: np.ndarray, data: dict, D: int, R: int,
                  districts: list, races: list) -> dict:
    """Posterior summaries from flattened post-burn samples (nsamp, n_params):
    per-cell thresholds, stop-weighted citywide thresholds per race, the race
    gap (races[1] - races[0], i.e. non-Black minus Black when races =
    ["black", "non_black_known_race"]), and posterior predictive errors."""
    p = unpack(chain, D, R)
    phi, lam, t = cell_params(p, D, R)                       # (nsamp, D, R)
    S, H = search_and_hit_rates(phi, lam, t)

    stops = data["stops"].astype(float)                      # (D, R)
    w = stops / np.clip(stops.sum(axis=0, keepdims=True), 1, None)
    t_city = (t * w[None, :, :]).sum(axis=1)                 # (nsamp, R)
    gap_samples = t_city[:, 1] - t_city[:, 0]                # non-Black - Black

    def summ(x):
        return {"mean": float(np.mean(x)),
                "ci_low": float(np.quantile(x, 0.025)),
                "ci_high": float(np.quantile(x, 0.975))}

    # Posterior predictive check. Absolute rate error alone punishes thin cells
    # for their own binomial noise, so the check is COVERAGE: for each cell,
    # draw replicated counts s* ~ Binom(stops, S) and h* ~ Binom(searches, H)
    # across posterior samples and ask whether the OBSERVED counts fall inside
    # the central 95% predictive interval. The posterior-mean deviations are
    # kept alongside for transparency.
    rng = np.random.default_rng(0)
    S_mean, H_mean = S.mean(axis=0), H.mean(axis=0)
    obs_S = np.divide(data["searches"], np.clip(stops, 1, None))
    obs_H = np.divide(data["hits"], np.clip(data["searches"], 1, None),
                      where=data["searches"] > 0,
                      out=np.zeros_like(S_mean))
    mask = stops > 0
    hmask = mask & (data["searches"] > 0)
    s_rep = rng.binomial(data["stops"][None, :, :], S)        # (nsamp, D, R)
    h_rep = rng.binomial(data["searches"][None, :, :], H)
    s_lo, s_hi = np.quantile(s_rep, [0.025, 0.975], axis=0)
    h_lo, h_hi = np.quantile(h_rep, [0.025, 0.975], axis=0)
    s_inside = (data["searches"] >= s_lo) & (data["searches"] <= s_hi)
    h_inside = (data["hits"] >= h_lo) & (data["hits"] <= h_hi)
    ppc = {
        "cells_checked": int(mask.sum()),
        "search_cells_outside_95pi": int((~s_inside & mask).sum()),
        "hit_cells_outside_95pi": int((~h_inside & hmask).sum()),
        "max_abs_search_rate_err": float(np.abs(S_mean - obs_S)[mask].max()),
        "max_abs_hit_rate_err": float(np.abs(H_mean - obs_H)[hmask].max()),
    }

    cells = []
    for di, d in enumerate(districts):
        for ri, r in enumerate(races):
            if data["stops"][di, ri] <= 0:
                continue
            cells.append({
                "district": d, "race": r,
                "stops": int(data["stops"][di, ri]),
                "searches": int(data["searches"][di, ri]),
                "hits": int(data["hits"][di, ri]),
                "threshold": summ(t[:, di, ri]),
                "search_rate_obs": float(obs_S[di, ri]),
                "hit_rate_obs": float(obs_H[di, ri]) if data["searches"][di, ri] else None,
            })
    return {
        "races": races,
        "citywide_thresholds": {r: summ(t_city[:, ri]) for ri, r in enumerate(races)},
        "gap": {**summ(gap_samples),
                "p_black_lower": float(np.mean(gap_samples > 0))},
        "cells": cells,
        "ppc": ppc,
    }
