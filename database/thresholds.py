"""Bayesian threshold test — stored-fit engine (read side).

THE TEST. Equal hit rates can hide discrimination: if the signal distributions
differ between groups, officers can apply a LOWER evidence bar to one group and
still find contraband at the same rate (the infra-marginality problem; Simoiu,
Corbett-Davies & Goel 2017). The threshold model infers the evidence bar
itself: for each district x race group it estimates the latent probability-of-
carrying threshold above which a search happens, from the observed stop,
search, and hit counts alone.

ARCHITECTURE. The model is fit OFFLINE (scripts/fit_threshold_test.py: MCMC via
emcee, minutes of compute) and the posterior summaries are persisted to the
``threshold_fits`` table; this module only READS the latest stored fit, so the
serving container needs no sampling dependencies and pages render in
milliseconds. Same degrade contract as every engine: RuntimeError when no fit
exists, so the surface omits itself instead of erroring.

The direction is significance-gated like every contested statistic on the
site: a race gap in thresholds is asserted only when the 95% posterior
interval of the gap excludes zero.
"""
from __future__ import annotations

import json
from typing import Any, Optional

MODEL_VERSION = "beta-signal-hier-v2"   # v2: additive race+district thresholds
                                        # (14 cells cannot identify per-cell ones)

THRESHOLD_DEFINITION = (
    "For each district and race group, the model treats a search as happening "
    "when the officer's assessed probability that the person carries contraband "
    "exceeds a latent threshold. Signal distributions (Beta, with district and "
    "race structure) and thresholds are inferred jointly from three observed "
    "counts per cell: stops, searches with a recorded outcome (consent, "
    "warrant, and protective pat-down searches; the source has no outcome field "
    "for probable-cause searches, which are counted and excluded), and searches "
    "where something was seized. A LOWER inferred threshold for one group means "
    "searches of that group proceed on thinner suspicion, evidence that equal "
    "hit rates alone cannot show. Citywide thresholds are stop-weighted "
    "averages across districts; the race gap is asserted only when its 95% "
    "posterior interval excludes zero."
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS threshold_fits (
    fit_id        BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    source_layer  INT NOT NULL,
    stop_batch    BIGINT NOT NULL,
    payload       JSONB NOT NULL
)
"""

_LATEST_SQL = """
SELECT fit_id, created_at, model_version, source_layer, stop_batch, payload
FROM threshold_fits ORDER BY fit_id DESC LIMIT 1
"""


async def ensure_table(conn) -> None:
    await conn.execute(_CREATE_SQL)


async def save_fit(conn, *, source_layer: int, stop_batch: int, payload: dict) -> int:
    await ensure_table(conn)
    return await conn.fetchval(
        "INSERT INTO threshold_fits (model_version, source_layer, stop_batch, payload) "
        "VALUES ($1, $2, $3, $4) RETURNING fit_id",
        MODEL_VERSION, source_layer, stop_batch, json.dumps(payload),
    )


async def latest_threshold_fit(conn) -> dict:
    """The latest stored fit, or RuntimeError when none exists (surface
    degrades, same contract as every engine)."""
    try:
        row = await conn.fetchrow(_LATEST_SQL)
    except Exception as exc:                      # UndefinedTableError included
        raise RuntimeError(f"threshold_fits unavailable: {exc}") from exc
    if row is None:
        raise RuntimeError(
            "no stored threshold fit — run scripts/fit_threshold_test.py first.")
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "fit_id": row["fit_id"],
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "source_layer": row["source_layer"],
        "stop_batch": row["stop_batch"],
        **payload,
    }


# ── Pure interpretation helpers (shared by page render and tests) ─────────────
def gap_direction(gap: dict) -> dict:
    """Direction of the race threshold gap, READ from the stored posterior.

    ``gap`` = {mean, ci_low, ci_high, p_black_lower} for
    (non-Black threshold - Black threshold): positive mean = Black people
    searched on a LOWER bar. Significant only when the 95% interval excludes 0.
    """
    lo, hi = gap.get("ci_low"), gap.get("ci_high")
    if lo is None or hi is None:
        return {"direction": "not comparable", "significant": False}
    if lo > 0:
        return {"direction": "black_lower", "significant": True}
    if hi < 0:
        return {"direction": "black_higher", "significant": True}
    return {"direction": "about the same", "significant": False}


def ppc_quality(ppc: dict) -> dict:
    """Whether the fitted model reproduces the observed counts (the posterior
    predictive check the model's credibility rests on).

    Primary check: coverage — every cell's observed search and hit counts must
    fall inside the central 95% posterior predictive interval, which accounts
    for each cell's own binomial noise (an absolute-error test would punish
    thin cells for their sample size). The posterior-mean rate deviations ride
    along for display."""
    s_out = ppc.get("search_cells_outside_95pi")
    h_out = ppc.get("hit_cells_outside_95pi")
    ok = s_out == 0 and h_out == 0
    return {"ok": ok,
            "search_cells_outside": s_out, "hit_cells_outside": h_out,
            "cells_checked": ppc.get("cells_checked"),
            "search_rate_err": ppc.get("max_abs_search_rate_err"),
            "hit_rate_err": ppc.get("max_abs_hit_rate_err")}
