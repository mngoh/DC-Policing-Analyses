"""Fit the Bayesian threshold test on the current stop batch and persist it.

Run LOCALLY (like the ingest scripts): needs scipy + emcee, which are
deliberately NOT deployment dependencies. The serving app reads the stored
posterior summaries from the ``threshold_fits`` table (database/thresholds.py).

    PYTHONPATH=. python scripts/fit_threshold_test.py [--steps 30000] [--dry-run]

Data per (district, race-group) cell, person-level, current layer-41 batch:
  stops     all persons stopped
  searches  persons with a consent / warrant / pat-down search AND a recorded
            seizure outcome (the outcome-bearing search types; probable-cause
            searches have no outcome field in the source and are excluded and
            counted, as on /searches)
  hits      searched persons where any recorded outcome shows a seizure
Unknown-race and unmapped-district persons are excluded and counted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emcee

from database.disparity import BLACK, _is_known_race, parse_seizure, resolve_current_batch
from database.thresholds import MODEL_VERSION, THRESHOLD_DEFINITION, save_fit
from scripts.threshold_model import (
    initial_ensemble,
    log_posterior,
    n_params,
    summarize_fit,
)

DISTRICTS = (1, 2, 3, 4, 5, 6, 7)
RACES = ("black", "non_black_known_race")
SOURCE_LAYER = 41

_CELLS_SQL = """
SELECT district, ethnicity,
       search_consent, search_warrant, pat_down, search_prob_cause,
       seized_consent_raw, seized_warrant_raw, seized_pat_down_raw,
       COUNT(*) AS n
FROM stops WHERE ingest_batch = $1
GROUP BY district, ethnicity, search_consent, search_warrant, pat_down,
         search_prob_cause, seized_consent_raw, seized_warrant_raw, seized_pat_down_raw
"""


def build_cells(rows: list[dict]) -> tuple[dict, dict]:
    """Person-level (D, R) count arrays + the loud exclusions block."""
    D, R = len(DISTRICTS), len(RACES)
    stops = np.zeros((D, R), dtype=np.int64)
    searches = np.zeros((D, R), dtype=np.int64)
    hits = np.zeros((D, R), dtype=np.int64)
    excl = {"unknown_race": 0, "unmapped_district": 0,
            "prob_cause_only_searches": 0, "outcome_unrecorded_searches": 0}

    for r in rows:
        n = r["n"]
        d = r["district"]
        if d is None or d not in DISTRICTS:
            excl["unmapped_district"] += n
            continue
        if not _is_known_race(r["ethnicity"]):
            excl["unknown_race"] += n
            continue
        di = DISTRICTS.index(d)
        ri = 0 if r["ethnicity"] == BLACK else 1
        stops[di, ri] += n

        outcome_searched = bool(r["search_consent"] or r["search_warrant"] or r["pat_down"])
        if not outcome_searched:
            if r["search_prob_cause"]:
                excl["prob_cause_only_searches"] += n
            continue
        outcomes = [parse_seizure(r["seized_consent_raw"]) if r["search_consent"] else None,
                    parse_seizure(r["seized_warrant_raw"]) if r["search_warrant"] else None,
                    parse_seizure(r["seized_pat_down_raw"]) if r["pat_down"] else None]
        recorded = [o for o in outcomes if o is not None]
        if not recorded:
            excl["outcome_unrecorded_searches"] += n
            continue
        searches[di, ri] += n
        if any(recorded):
            hits[di, ri] += n

    return {"stops": stops, "searches": searches, "hits": hits}, excl


def fit(data: dict, *, steps: int, seed: int = 20260730) -> tuple[dict, dict]:
    D, R = len(DISTRICTS), len(RACES)
    ndim = n_params(D, R)
    nwalkers = max(4 * ndim, 96)
    rng = np.random.default_rng(seed)
    p0 = initial_ensemble(rng, nwalkers, D, R, data)

    # Differential-evolution moves mix far better than the default stretch move
    # on this correlated 38-dim posterior (default StretchMove gave tau ~ 2000).
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(data, D, R), vectorize=True,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    t0 = time.time()
    sampler.run_mcmc(p0, steps, progress=True)
    elapsed = time.time() - t0

    burn = steps // 2
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        tau_max = float(np.nanmax(tau))
    except Exception:
        tau_max = float("nan")
    thin = max(1, int(tau_max // 2)) if np.isfinite(tau_max) else 10
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    # emcee's own reliability rule: the post-burn chain must be at least 50
    # autocorrelation times long. An unconverged fit is still persisted (for
    # the record) but the page will refuse to assert a direction from it.
    converged = bool(np.isfinite(tau_max) and (steps - burn) >= 50 * tau_max)
    diagnostics = {
        "sampler": "emcee affine-invariant ensemble (DE moves)",
        "walkers": nwalkers, "steps": steps, "burn": burn, "thin": thin,
        "posterior_samples": int(chain.shape[0]),
        "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        "max_autocorr_time": tau_max,
        "converged": converged,
        "seconds": round(elapsed, 1),
        "seed": seed,
    }
    summary = summarize_fit(chain, data, D, R, list(DISTRICTS), list(RACES))
    return summary, diagnostics


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and print, do not persist")
    args = ap.parse_args()

    import asyncpg
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is required")
    conn = await asyncpg.connect(url, timeout=30)
    try:
        batch = await resolve_current_batch(conn, SOURCE_LAYER)
        if batch is None:
            raise SystemExit("no finished stop batch for layer 41")
        rows = [dict(r) for r in await conn.fetch(_CELLS_SQL, batch)]
        data, excl = build_cells(rows)
        print(f"batch {batch}: stops {int(data['stops'].sum()):,}, "
              f"searches {int(data['searches'].sum()):,}, "
              f"hits {int(data['hits'].sum()):,}; exclusions {excl}")

        summary, diagnostics = fit(data, steps=args.steps, seed=args.seed)
        payload = {
            "definition": THRESHOLD_DEFINITION,
            "model_version": MODEL_VERSION,
            "exclusions": excl,
            "diagnostics": diagnostics,
            **summary,
        }
        print(json.dumps({k: payload[k] for k in
                          ("citywide_thresholds", "gap", "ppc", "diagnostics")},
                         indent=2))
        if args.dry_run:
            print("(dry run: not persisted)")
            return
        fit_id = await save_fit(conn, source_layer=SOURCE_LAYER,
                                stop_batch=batch, payload=payload)
        print(f"persisted threshold fit_id={fit_id}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
