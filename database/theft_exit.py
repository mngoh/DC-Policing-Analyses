"""District-level theft-exit analysis: pre-registered design and finding
language for /classification/districts.

PRE-REGISTRATION. This module deliberately contains NO estimator. It fixes,
in version control, before any estimate exists:
  * the design constants (treated cells, windows, exclusions, outcome),
  * the a-priori exclusion of tests that cannot detect at documented scale,
  * the finding language for every branch the estimate could land in, and
  * the branch-selection rules that map a result to that language.
The estimator lands in a later commit. Git history is the receipt: the
interpretation of every possible outcome was written down first, so no
wording can be tuned to the number it describes.

DESIGN. A pooled stacked difference-in-differences event study on monthly
theft-category counts (THEFT/OTHER + THEFT F/AUTO, log scale), one stack per
(district, window) pair that MPD's internal affairs report ties to
documented reclassification (Final Investigative Report IS #26-000051,
March 5, 2026; page cites on each cell below). Controls are the remaining
district-months, minus a documented-contamination exclusion (3D, see
CONTROL_EXCLUSION). Inference is by permutation across placebo district
assignments, because cluster asymptotics are meaningless at seven districts.
Pre-window lead coefficients are a first-class output: if treated and
control districts were not parallel before a window opened, that cell's
estimate is an artifact and the pre-registered language below says so.

RESULT SCHEMA (produced by the estimator commit; the text functions and
their tests are fixed against this shape first):
  result = {
    "pooled": {
        "beta": float,            # log-point deficit; negative = fewer thefts
        "p_perm": float,          # one-sided permutation p (deficit direction)
        "n_placebo": int,
        "discovery": bool,        # beta < 0 and p_perm <= 0.05
        "weights": {cell_key: float},   # treated-observation share per cell
    },
    "cells": [{
        "key": str, "district": str, "label": str,
        "beta": float, "se_nw": float,
        "p_perm": float, "p_adjusted": float,   # BH within the family of 3
        "discovery": bool,
        "pretrend": {"joint_p": float, "ok": bool},
        "bridge": {"doc_per_month": float, "base_per_month": float,
                   "expected_dip_pct": float, "mde_pct": float,
                   "below_floor": bool},
        "event_path": [{"month": date, "rel": int, "coef": float}],
        # anchor cell only: the pre-registered THRIVE timing call
        "thrive_timing": {"pre_thrive_mean": float, "window_mean": float,
                          "opens_before": bool},
    }],
    "sensitivity_3d": {"beta": float, "p_perm": float, "discovery": bool},
    "backfill": {"recovered_rows": int},
  }
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Optional, Sequence

from database.classification_mix import (SURGE_START, IA_REPORT_CITE,
                                         _mat_inv, _mat_solve, add_months,
                                         benjamini_hochberg, month_range)

# ── Outcome ───────────────────────────────────────────────────────────────────
# The documented originals span "Theft I, Theft II, Theft From Auto, Theft
# Stolen Auto, etc." (p. 217), so the outcome is the combined theft-category
# count. Motor vehicle theft is excluded a-priori: the Kia/Hyundai wave and
# its 2024 collapse would dominate any district contrast, and the documented
# stolen-auto edits number about nine (p. 533). THEFT/OTHER alone is run as
# a sensitivity, not a second primary.
OUTCOME_OFFENSES = ("THEFT/OTHER", "THEFT F/AUTO")
OUTCOME_SENSITIVITY = ("THEFT/OTHER",)

# ── Treated cells, from the report's own per-district statements ──────────────
# Attribution rule: a cell exists only where the report itself names the
# district and a window. Editors crossed district lines (2D captains editing
# 5D THRIVE reports, pp. 547-548; seven 1D edits by a 7D THRIVE captain,
# pp. 538, 553), so "editor's command" is never used to place edits.
# doc_per_month is the pre-registered documented signal floor for the cell's
# magnitude bridge; the base volume comes from data at estimation time.
CELLS: tuple[dict[str, Any], ...] = (
    {"key": "7D-2024", "district": "7",
     "label": "Seventh District, January-October 2024",
     "start": date(2024, 1, 1), "end_incl": date(2024, 10, 1),
     "docs": (("131 reports reclassified; audit window Jan 25 - Oct 8, 2024",
               "p. 217"),
              ("~20 reports reclassified in the Seventh District", "p. 546"),
              ("commander-directed misclassification, documented but "
               "uncounted", "pp. 88-89")),
     "doc_per_month": 15.1},          # (131 + 20) / 10 window months
    {"key": "2D-2025", "district": "2",
     "label": "Second District, January-August 2025",
     "start": date(2025, 1, 1), "end_incl": date(2025, 8, 1),
     "docs": (("~299 reports improperly reclassified in 2D and the 5D "
               "THRIVE area (split not quantified)", "p. 547"),
              ("~151 reports improperly reclassified in 2D and the 5D "
               "THRIVE area (split not quantified)", "p. 548"),
              ("123 of 156 2D reports classified TPWR Jan-Aug 2025 flagged "
               "as misclassified by the report's own review", "p. 317")),
     "doc_per_month": 15.4},          # 123 flagged / 8 window months
    {"key": "5D-2025", "district": "5",
     "label": "Fifth District (Rosedale THRIVE), February-August 2025",
     "start": date(2025, 2, 1), "end_incl": date(2025, 8, 1),
     "docs": (("Rosedale THRIVE lead assignments spanning the window",
               "p. 14"),
              ("59 of 68 5D reports classified TPWR in 2025 flagged as "
               "misclassified by the report's own review", "p. 317"),
              ("5D TPWR uptick coinciding with the THRIVE leads",
               "pp. 532-533")),
     "doc_per_month": 7.4},           # 59 flagged / 8 months (Jan-Aug basis)
)
ANCHOR_KEY = "7D-2024"   # the only cell whose documented scale clearly
                         # exceeds a detection floor (feasibility, 2026-07)

# ── Stack geometry ────────────────────────────────────────────────────────────
STACK_PRE_MONTHS = 18
STACK_POST_MONTHS = 18
# Every month at or past the federal surge boundary is dropped from every
# stack, treated and control alike. Consequence, stated rather than hidden:
# the 2D and 5D stacks have essentially no post-window leg, and the pooled
# coefficient is in practice mostly the 7D 2024 cell (see
# composition_sentence).

# ── Control-pool exclusion (3D), pre-registered as primary ────────────────────
# The report documents 157 reclassifications by the then-3D commander (p. 16)
# but gives no audit date range; "six months" is one witness's
# characterization (p. 71) and the discovery date is May 12, 2025 (p. 15).
# Because the window is inferred, the exclusion is deliberately over-wide:
# roughly twice the characterized span plus margin, June 2024 through the end
# of the estimation span. The estimate is reported both ways (3D excluded is
# primary; 3D included is the sensitivity), and sensitivity_sentence() fixes
# in advance what each combination of outcomes is allowed to claim.
CONTROL_EXCLUSION: dict[str, Any] = {"district": "3",
                                     "start": date(2024, 6, 1),
                                     "end_excl": SURGE_START}

# ── THRIVE spans per treated district (roster, pp. 13-14) ─────────────────────
# THRIVE is a competing explanation with the same signature as
# reclassification: a targeted initiative that genuinely reduced theft would
# also leave a district-shaped deficit against controls. These spans are
# drawn on every event-study plot so the reader can see whether a deficit
# tracks the audit window or the initiative window.
THRIVE_SPANS: dict[str, tuple[dict[str, Any], ...]] = {
    "7": ({"area": "Anacostia", "start": date(2024, 5, 2),
           "end": date(2025, 9, 3), "cite": "pp. 13-14"},
          {"area": "Washington Highlands", "start": date(2024, 9, 5),
           "end": date(2025, 1, 2), "cite": "pp. 13-14"},
          {"area": "Washington Highlands", "start": date(2025, 5, 2),
           "end": date(2025, 9, 3), "cite": "p. 14"},
          {"area": "Buena Vista", "start": date(2025, 1, 3),
           "end": date(2025, 9, 3), "cite": "p. 14"}),
    "5": ({"area": "Rosedale", "start": date(2025, 1, 3),
           "end": date(2025, 9, 3), "cite": "p. 14"},),
    "2": (),   # no THRIVE area operated in 2D; its documented mechanism is
               # district-level editing under command direction (pp. 547-548)
}

# ── Statistical conventions ───────────────────────────────────────────────────
PERMUTATION_ALPHA = 0.05
PRETREND_ALPHA = 0.10   # joint lead test; deliberately stricter-than-usual
                        # gate (a marginal pre-trend kills the cell rather
                        # than surviving on the benefit of the doubt)
BH_Q = 0.05
PRETREND_LEADS = 12     # lead coefficients entering the joint statistic
# The 7D deficit "opens before" the THRIVE boundary if at least half the
# estimated window deficit is already present, on average, in the window
# months that predate the first THRIVE area in the district (Jan-Apr 2024;
# Anacostia begins May 2, 2024, p. 13). Fixed here, before estimation, so
# the timing call cannot be tuned to the path it describes.
THRIVE_OPENS_BEFORE_FRACTION = 0.5
ANCHOR_PRE_THRIVE_MONTHS = (date(2024, 1, 1), date(2024, 2, 1),
                            date(2024, 3, 1), date(2024, 4, 1))

# ══════════════════════════════════════════════════════════════════════════════
# Pre-registered language. Everything below is wording, fixed before any
# estimate exists. Formatting placeholders take live numbers; no sentence
# changes shape based on which way a number lands, only the pre-registered
# branch selection chooses among them.
# ══════════════════════════════════════════════════════════════════════════════

NONALIGNMENT_SENTENCE = (
    "This data sits inside a live dispute between federal and local "
    "officials over DC crime. This analysis does not adjudicate that "
    "dispute; it reports what the published data can and cannot show, with "
    "the confounds labeled.")

APRIORI_EXCLUSION = (
    "Two families of tests were excluded before any estimate was computed: "
    "per-district robbery-share contrasts and per-district ADW-share "
    "contrasts. The internal affairs report's documented robbery-path and "
    "ADW-path reclassifications total roughly 5 and 11 reports respectively "
    "(pp. 16, 256, 309); at every district's denominator that scale sits "
    "one to two orders of magnitude below any detection floor. Running "
    "those tests could only produce uninformative nulls, and reporting "
    "uninformative nulls alongside real results would overstate what was "
    "checked. The exclusion, and the finding language for every branch "
    "below, were committed to version control before the estimator existed.")

PRETREND_INFERENCE_TEXT = (
    "The pre-trend gate uses the same inference as the estimate itself. "
    "With seven districts, asymptotic joint tests have unknown calibration, "
    "and this gate can kill the entire analysis, so its p-value cannot rest "
    "on an approximation the primary test already rejects. The joint lead "
    "statistic is therefore recomputed under the same placebo procedure as "
    "the primary estimate: the treated label is reassigned across "
    "comparison districts and across placebo window placements, the "
    "statistic is recomputed for each assignment, and the gate's p-value "
    "is the rank of the real statistic in that placebo distribution.")

EXCLUSION_3D_TEXT = (
    "The Third District is excluded from the comparison pool from June 2024 "
    "through the end of the estimation span. The report documents 157 "
    "reclassifications by the then-Third District commander (p. 16) but "
    "gives no audit date range; the six-month span comes from one witness's "
    "characterization (p. 71) and the May 12, 2025 discovery date (p. 15). "
    "Because the window is inferred, the exclusion is deliberately "
    "over-wide, roughly twice the characterized span plus margin, so that "
    "no plausibly contaminated Third District month anchors the comparison. "
    "The estimate is reported both with this exclusion (primary) and "
    "without it (sensitivity).")

# The confound block. Every sentence travels with every branch; none is
# demoted to a footnote. THRIVE sits at equal prominence with the citywide
# decline and the contaminated controls, per the pre-registered design.
CONFOUND_SENTENCES: tuple[str, ...] = (
    # citywide decline
    "Citywide theft fell roughly 14% in 2024, so a raw dip in any one "
    "district means nothing; the comparison-based design carries the "
    "entire argument.",
    # THRIVE as a competing explanation with the same signature
    "THRIVE, a targeted enforcement initiative, operated in the Fifth and "
    "Seventh District areas during these windows, and an initiative that "
    "genuinely reduced theft would leave the same district-shaped deficit "
    "as reports edited out of the feed. This data cannot separate those "
    "two readings; the event-study plots mark both the audit windows and "
    "the THRIVE spans so the timing can be compared, and the finding text "
    "carries the ambiguity rather than resolving it.",
    # contaminated controls
    "The comparison districts are not certifiably clean: a whistleblower "
    "alleged all seven districts, and the report calls its own tally 'by "
    "no means a complete record' (p. 526). Contamination of controls "
    "biases this test toward finding nothing, so a positive result is a "
    "floor-style claim, at least this much, and never a clean causal "
    "measurement.",
    # attribution rule
    "Attribution of edits to districts comes from the report's own "
    "per-district statements, never from an editor's assignment: editors "
    "crossed district lines (pp. 538, 547-548, 553).",
    # surge boundary
    "Nothing at or after the August 2025 federal surge boundary is used, "
    "in any district, treated or control.",
    # TPWR exhibit
    "The report's TPWR-by-district table (pp. 317-318) cannot be "
    "reproduced from public data, because TPWR never enters the open "
    "feed; where it appears here it is presented as the report's own "
    "exhibit, cited, not as something this page verified.",
    # boundary churn
    "District boundaries are stable across the estimation span (block-level "
    "churn 0.6-0.9% a year, all post-2012-realignment); no series here "
    "extends earlier.",
)

THRIVE_TIMING_NOTE = (
    "The Seventh District audit window opens in January 2024, about three "
    "months before the first THRIVE area in that district (Anacostia, "
    "May 2, 2024, p. 13), and the district-wide property-crime remediation "
    "directives it rests on predate THRIVE (January 2024 emails, p. 218). "
    "If the deficit is present in those pre-THRIVE months, the initiative "
    "reading weakens for that cell. The Fifth District window coincides "
    "almost exactly with the Rosedale THRIVE span (p. 14), so for that "
    "cell the two readings are inseparable by timing. No THRIVE area "
    "operated in the Second District.")

ENFORCEMENT_EXHIBIT_TEXT = (
    "District-level arrest and stop volumes are shown alongside the event "
    "study as context. They can show whether visible enforcement activity "
    "changed during the windows; they cannot rule the initiative reading "
    "out, because a targeted initiative can reduce theft through presence "
    "and deterrence without moving arrests or stops. No exhibit here "
    "separates the two readings; where the text says the readings are "
    "inseparable, that is a statement about this data, not a gap left "
    "unexamined.")

DID_NULL_SCOPE = (
    "A null here does not mean the manipulation did not happen; it is "
    "documented at case level in sustained IAD findings. It means the "
    "documented theft-exit pattern is not visible in public monthly "
    "aggregates even where it should have been.")

PLAIN_QUESTION = (
    "Where MPD's own investigators documented theft reports being "
    "reclassified out of the public feed, is the hole visible from "
    "outside?")


# ── Formatting helpers ────────────────────────────────────────────────────────
def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}" if p < 0.01 else f"= {p:.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.0f}%" if abs(x) >= 10 else f"{x:.1f}%"


def _beta_pct(beta: float) -> float:
    """Log-point coefficient as a percent deficit (positive number)."""
    return (1.0 - math.exp(beta)) * 100.0


def cell_by_key(result: dict, key: str) -> Optional[dict]:
    for c in result.get("cells", []):
        if c.get("key") == key:
            return c
    return None


# ── Branch selection (pre-registered rules) ───────────────────────────────────
def active_cells(result: dict) -> list[dict]:
    """Cells whose pre-trends hold. Cells that fail pre-trends are excluded
    from the pooled estimate and carry the pre-trend-failure text; they never
    contribute evidence in either direction."""
    return [c for c in result["cells"] if c["pretrend"]["ok"]]


def did_outcome(result: dict) -> str:
    """Which pre-registered branch the result is in.

    'design_fails': the anchor cell (7D-2024, the only cell whose documented
        scale clearly exceeds a detection floor) fails pre-trends. The
        remaining cells cannot detect, so the design cannot answer the
        question in either direction, regardless of what fires elsewhere.
    'consistent':  the pooled test and the anchor cell both fire.
    'null':        neither the pooled test nor any pre-trend-passing cell
        fires.
    'mixed':       everything else. Reported as mixed; nothing is headlined.
    """
    anchor = cell_by_key(result, ANCHOR_KEY)
    if anchor is None or not anchor["pretrend"]["ok"]:
        return "design_fails"
    pooled_fired = bool(result["pooled"]["discovery"])
    anchor_fired = bool(anchor["discovery"])
    if pooled_fired and anchor_fired:
        return "consistent"
    if not pooled_fired and not any(c["discovery"] for c in active_cells(result)):
        return "null"
    return "mixed"


def sensitivity_outcome(result: dict) -> str:
    """How the 3D-exclusion sensitivity interacts with the primary result.

    'both':          primary fires and so does the 3D-included run.
    'primary_only':  primary fires, 3D-included does not; the finding is
        correspondingly weaker and the page says so.
    'na_null':       primary does not fire. The sensitivity is reported for
        completeness but under the pre-registered rules it cannot upgrade a
        primary null into a finding.
    """
    primary = bool(result["pooled"]["discovery"])
    sens = bool(result.get("sensitivity_3d", {}).get("discovery"))
    if not primary:
        return "na_null"
    return "both" if sens else "primary_only"


# ── Pre-registered finding language ───────────────────────────────────────────
def pretrend_sentence(result: dict) -> str:
    """Pre-trends are a first-class output. Clean pre-trends are stated
    prominently; a failure is stated per cell by cell_finding_text and, for
    the anchor, escalates to the design_fails branch."""
    cells = result["cells"]
    failed = [c for c in cells if not c["pretrend"]["ok"]]
    if not failed:
        worst = max(c["pretrend"]["joint_p"] for c in cells)
        return (
            "Before each documented window opens, the treated district and "
            "the comparison districts show no visible divergence: the joint "
            "test that all pre-window lead coefficients are zero does not "
            "reject in any stack (smallest margin, permutation joint p "
            f"{_fmt_p(worst)}). Parallel pre-trends are a necessary "
            "condition for this design, and it passes that gate. Passing "
            "is absence of visible violation, not proof of parallelism; "
            "the test's power is limited, so the full lead-and-lag path "
            "is plotted and parallelism should be judged by eye, not on a "
            "single p-value.")
    names = ", ".join(c["label"] for c in failed)
    return (
        f"Pre-trends fail for {names}: the joint test that pre-window lead "
        "coefficients are zero rejects there. Those cells are excluded from "
        "the pooled estimate and carry no evidentiary weight in either "
        "direction; the cells that remain are the only ones read.")


def cell_finding_text(cell: dict) -> str:
    """The per-cell sentence, chosen by pre-registered rules only."""
    label = cell["label"]
    b = cell["bridge"]
    if not cell["pretrend"]["ok"]:
        return (
            f"Pre-trends fail for this cell: the {label} series was already "
            "diverging from the comparison districts before the documented "
            "window opened (joint lead test p "
            f"{_fmt_p(cell['pretrend']['joint_p'])}). The design does not "
            "hold here. This cell is excluded from the pooled estimate, and "
            "no finding, in either direction, is drawn from it: a deficit "
            "measured against a comparison the district was not tracking "
            "beforehand would be an artifact of the comparison, not "
            "evidence about classification.")
    if b["below_floor"]:
        return (
            f"The documented scale for this cell, roughly "
            f"{b['doc_per_month']:.0f} reports a month against a base of "
            f"roughly {b['base_per_month']:.0f} a month "
            f"({'; '.join(p for _, p in cell['docs'])}), sits below this "
            f"cell's detection floor of {_fmt_pct(b['mde_pct'])}. This cell "
            "cannot confirm or rule out the documented pattern; a null here "
            "is uninformative by construction and is shown for "
            "completeness, not as evidence in either direction.")
    if cell["discovery"]:
        return (
            f"In the {label} window, theft-category counts run "
            f"{_fmt_pct(_beta_pct(cell['beta']))} below the comparison "
            f"districts' contemporaneous pattern (one-sided permutation p "
            f"{_fmt_p(cell['p_perm'])}, BH-adjusted p "
            f"{_fmt_p(cell['p_adjusted'])}), against a documented floor of "
            f"roughly {b['doc_per_month']:.0f} reports a month "
            f"({'; '.join(p for _, p in cell['docs'])}).")
    return (
        f"In the {label} window, theft-category counts do not run "
        "measurably below the comparison districts' pattern (one-sided "
        f"permutation p {_fmt_p(cell['p_perm'])}), even though the "
        f"documented scale here, roughly {b['doc_per_month']:.0f} reports a "
        f"month against a base of roughly {b['base_per_month']:.0f}, "
        f"exceeds this cell's detection floor of {_fmt_pct(b['mde_pct'])}. "
        "This is the informative kind of null: the documented pattern "
        "should have been visible here and is not.")


def thrive_timing_sentence(result: dict) -> str:
    """Pre-registered timing call for the anchor cell: does the 7D deficit
    open before or at/after the first THRIVE area in the district? Written,
    both directions, before estimation. Rendered only when the anchor cell
    shows a deficit; timing has nothing to separate otherwise. The selection
    rule is THRIVE_OPENS_BEFORE_FRACTION, fixed above.
    """
    anchor = cell_by_key(result, ANCHOR_KEY)
    if anchor is None or not anchor["pretrend"]["ok"] or not anchor["discovery"]:
        return ("With no measurable deficit in the Seventh District cell, "
                "the window-versus-THRIVE timing comparison has nothing to "
                "separate, and no timing claim is made.")
    timing = anchor.get("thrive_timing") or {}
    if timing.get("opens_before"):
        return (
            "The Seventh District deficit opens before the THRIVE boundary: "
            "the January-April 2024 window months, which predate the first "
            "THRIVE area in the district (Anacostia, May 2, 2024, p. 13), "
            "already carry at least half the estimated window deficit. "
            "Those months are covered only by the documented audit window "
            "(p. 217) and the district-wide remediation directives that "
            "predate THRIVE (p. 218), which weakens the initiative reading "
            "for this cell. It does not eliminate it; enforcement can "
            "change in ways no roster marks.")
    return (
        "The Seventh District deficit opens at or after the THRIVE "
        "boundary: the January-April 2024 window months, which predate the "
        "first THRIVE area in the district (Anacostia, May 2, 2024, p. 13), "
        "carry less than half the estimated window deficit. The timing "
        "does not separate the documented-reclassification reading from "
        "the initiative reading for this cell, and the finding text "
        "carries both.")


def composition_sentence(result: dict) -> str:
    """State the pooled estimate's real composition. 'Pooled across three
    cells' must not imply three independent replications."""
    w = result["pooled"]["weights"]
    anchor_w = w.get(ANCHOR_KEY, 0.0)
    others = [(k, v) for k, v in sorted(w.items()) if k != ANCHOR_KEY]
    others_s = " and ".join(f"{_fmt_pct(v * 100)} ({k})" for k, v in others)
    return (
        "'Pooled across three documented cells' overstates the pool: with "
        "every month from August 2025 onward excluded, the other stacks "
        f"contribute {others_s} of the treated observations, and the pooled "
        f"coefficient is in practice mostly the Seventh District 2024 cell "
        f"({_fmt_pct(anchor_w * 100)}). Read the pooled number as one "
        "well-documented cell plus two thin corroborating ones, not three "
        "independent replications.")


def sensitivity_sentence(result: dict) -> str:
    out = sensitivity_outcome(result)
    if out == "both":
        return (
            "The estimate survives both ways: with the Third District "
            "excluded from the comparison pool (primary) and with it "
            "included (sensitivity). That is robustness the design gets "
            "for free, and it removes the exclusion as a load-bearing "
            "choice.")
    if out == "primary_only":
        return (
            "The estimate survives only when the Third District is excluded "
            "from the comparison pool. The exclusion is justified, the "
            "report documents 157 reclassifications by that district's "
            "then-commander in a window inferred from witness testimony "
            "(pp. 16, 71), but an estimate that depends on a justified "
            "exclusion is weaker than one that does not, and this page "
            "treats it that way.")
    return (
        "The sensitivity run with the Third District included in the "
        "comparison pool is reported for completeness; under the "
        "pre-registered rules it cannot upgrade a primary null into a "
        "finding.")


def did_finding_text(result: dict) -> str:
    """The page's top-level finding paragraph, selected by did_outcome."""
    outcome = did_outcome(result)
    pooled = result["pooled"]
    anchor = cell_by_key(result, ANCHOR_KEY)

    if outcome == "design_fails":
        return (
            "This design cannot answer the question. The Seventh District "
            "2024 window is the only cell whose documented scale exceeds a "
            "detection floor, and its pre-trends fail: the district was not "
            "moving in parallel with the comparison districts before the "
            "audit window opened (joint lead test p "
            f"{_fmt_p(anchor['pretrend']['joint_p']) if anchor else '?'}), "
            "so a difference-in-differences estimate there would be an "
            "artifact of the comparison. The remaining cells sit below "
            "their detection floors, so nothing that fires or fails to "
            "fire in them can carry a claim. The honest output of this "
            "page is that answer, not a weaker substitute test.")

    if outcome == "consistent":
        assert anchor is not None
        return (
            "Yes, where the documented scale was large enough to see. "
            "Theft-category counts in the district-months where the "
            "internal affairs report documents concentrated "
            f"reclassification run {_fmt_pct(_beta_pct(pooled['beta']))} "
            "below the comparison districts' contemporaneous pattern "
            "(pooled stacked difference-in-differences, one-sided "
            f"permutation p {_fmt_p(pooled['p_perm'])}), and the deficit is "
            "present in the one cell whose documented scale clearly "
            "exceeds this test's detection floor: the Seventh District "
            "during the January-October 2024 audit window (p. 217). This "
            "is not proof of manipulation and is not offered as such; the "
            "manipulation is already documented at case level in sustained "
            "IAD findings. What this test adds is independent: the hole "
            "those findings describe is visible in the public feed, at a "
            "scale at least as large as the documented floor. Two readings "
            "of the deficit remain open, and this test cannot separate "
            "them: theft reports moved out of the public feed, as the "
            "sustained findings document, or a targeted enforcement "
            "initiative genuinely reducing theft in the same places over "
            "the same months. THRIVE operated in the Fifth and Seventh "
            "District areas during these windows, and an initiative that "
            "works leaves the same district-shaped hole as an initiative "
            "whose numbers were edited; the event-study plots mark both "
            "the audit windows and the THRIVE spans, and where they "
            "coincide the two readings are inseparable in this data. "
            "Because the comparison districts are not certifiably clean, "
            "the report calls its own tally 'by no means a complete "
            "record' (p. 526), this estimate is a floor on the documented "
            "pattern, not a measurement of its total size.")

    if outcome == "null":
        assert anchor is not None
        ab = anchor["bridge"]
        return (
            "No. Theft-category counts in the documented district-windows "
            "do not run measurably below the comparison districts' pattern "
            f"(pooled one-sided permutation p {_fmt_p(pooled['p_perm'])}), "
            "including in the Seventh District 2024 cell, where the "
            f"documented scale, roughly {ab['doc_per_month']:.0f} reports "
            f"a month against a base of roughly "
            f"{ab['base_per_month']:.0f} a month (pp. 217, 546), exceeds "
            f"this test's detection floor. {DID_NULL_SCOPE} That is a "
            "finding about the public feed as an oversight instrument: "
            "case-level manipulation of this documented scale can pass "
            "through it without leaving a detectable aggregate trace, "
            "which is one reason internal audit access, not public data, "
            "is what surfaced the conduct.")

    # mixed
    assert anchor is not None
    pooled_fired = bool(pooled["discovery"])
    if pooled_fired and not anchor["discovery"]:
        shape = (
            "The pooled test fires but the Seventh District 2024 cell, the "
            "only cell whose documented scale clearly exceeds its "
            "detection floor, does not.")
    elif anchor["discovery"] and not pooled_fired:
        shape = (
            "The Seventh District 2024 cell fires but the pooled test "
            "does not.")
    else:
        fired = [c["label"] for c in active_cells(result) if c["discovery"]]
        shape = (
            f"A below-floor cell fires ({', '.join(fired)}) while the "
            "pooled test and the one above-floor cell do not; a discovery "
            "in a cell that cannot detect the documented scale is exactly "
            "the pattern noise produces.")
    return (
        f"Mixed, and reported as mixed. {shape} Both facts carry equal "
        "weight here; neither alone supports the headline claim, so this "
        "page does not make it. A deficit concentrated outside the "
        "best-documented cell is consistent with the report's caveat that "
        "its tally is not complete (p. 526), and equally consistent with "
        "ordinary district-level variation; the placebo distribution shown "
        "below is the calibration for how often comparison districts "
        "produce swings of this size on their own.")


def did_plain_answer(result: dict) -> str:
    """The short plain-language answer box."""
    outcome = did_outcome(result)
    if outcome == "design_fails":
        return ("No answer is possible with this design: the only cell "
                "large enough to test was not moving in parallel with the "
                "comparison districts beforehand, so the comparison would "
                "be an artifact. The page shows why instead of substituting "
                "a weaker test.")
    if outcome == "consistent":
        return ("Yes, with the limits stated. Where the internal affairs "
                "report documents concentrated reclassification at a scale "
                "this test can see, public theft counts run below the "
                "comparison districts' pattern. That is consistency with "
                "the documented record, not independent proof, and the "
                "same deficit is equally consistent with the THRIVE "
                "enforcement initiative genuinely reducing theft; the two "
                "readings are not separable here.")
    if outcome == "null":
        return ("No. " + DID_NULL_SCOPE + " Public data alone would not "
                "have caught this, which is itself the finding.")
    return ("Mixed, and the page says so rather than headlining the part "
            "that fired. Neither the pooled estimate nor the "
            "best-documented cell carries the claim alone.")


# ══════════════════════════════════════════════════════════════════════════════
# Estimator. Added in a separate commit AFTER the pre-registered language
# above; the git history is the receipt. Everything here is design-based:
# the only p-values are permutation ranks in placebo ensembles, computed by
# the identical procedure for the estimate and for the pre-trend gate.
# ══════════════════════════════════════════════════════════════════════════════

DISTRICTS = ("1", "2", "3", "4", "5", "6", "7")
MIN_PLACEBO_PRE_MONTHS = PRETREND_LEADS + 1   # 12 estimable leads PLUS at
                              # least one baseline month before them; with
                              # zero baseline months the treated district's
                              # fixed effect is collinear with its event path
MIN_PLACEBO_LEADS = 6         # at least 6 present lead months
# scripts/backfill_district.py (point-in-polygon on the PostGIS location
# column) ran 2026-07-31, BEFORE any estimation: all 1,564 rows missing a
# district were recovered (336 of them a single Nov 2024 ingest batch);
# zero null-district rows remain. The count is reported on the page.
BACKFILL_RECOVERED_ROWS: Optional[int] = 1564


def _mdiff(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _is_masked(district: str, month: date, exclude_3d: bool) -> bool:
    """District-months that may never serve as controls: every cell's own
    documented window, plus the over-wide 3D exclusion (primary only)."""
    if (exclude_3d and district == CONTROL_EXCLUSION["district"]
            and CONTROL_EXCLUSION["start"] <= month
            < CONTROL_EXCLUSION["end_excl"]):
        return True
    for cell in CELLS:
        if district == cell["district"] and cell["start"] <= month <= cell["end_incl"]:
            return True
    return False


def _stack_span(cell: dict) -> list[date]:
    lo = add_months(cell["start"], -STACK_PRE_MONTHS)
    hi = min(add_months(cell["end_incl"], STACK_POST_MONTHS + 1), SURGE_START)
    return month_range(lo, hi)


def _estimable_window(cell: dict) -> list[date]:
    return [m for m in month_range(cell["start"], add_months(cell["end_incl"], 1))
            if m < SURGE_START]


def _fit_stack(counts: dict[str, dict[date, int]], cell: dict,
               treated: str, t_start: date, exclude_3d: bool) -> Optional[dict]:
    """Event-study OLS for one stack under one (possibly placebo) treated
    assignment: month dummies + district dummies + event-time dummies on the
    treated district (rel -PRETREND_LEADS .. span end; earlier months are the
    baseline). Returns the event path, the window-mean coefficient, and the
    lead statistic (mean squared standardized lead), or None if the
    assignment is infeasible."""
    months = _stack_span(cell)
    L = len(_estimable_window(cell))
    win = set(month_range(t_start, add_months(t_start, L)))
    if not all(m in set(months) for m in win):
        return None
    idx_of = {m: i for i, m in enumerate(months)}
    if idx_of[t_start] < MIN_PLACEBO_PRE_MONTHS:
        return None

    # Rows: (dummy index list, y). A row enters iff it is unmasked, or it is
    # the treated district inside the assignment window (which, for the real
    # assignment, is exactly the documented window).
    raw: list[tuple[str, date, float]] = []
    for d in DISTRICTS:
        dm = counts.get(d, {})
        for m in months:
            n = dm.get(m, 0)
            if n <= 0:
                continue
            in_win = d == treated and m in win
            if _is_masked(d, m, exclude_3d) and not in_win:
                continue
            raw.append((d, m, math.log(float(n))))
    if treated not in {d for d, _, _ in raw}:
        return None
    # Placebo windows must be fully present for the treated district.
    present_treated = {m for d, m, _ in raw if d == treated}
    if not win <= present_treated:
        return None
    rels_present = sorted({_mdiff(t_start, m) for m in present_treated
                           if _mdiff(t_start, m) >= -PRETREND_LEADS})
    leads_present = [r for r in rels_present if r < 0]
    if len(leads_present) < MIN_PLACEBO_LEADS:
        return None
    # The treated district needs baseline months (rel < -PRETREND_LEADS);
    # without them its fixed effect is collinear with the event dummies.
    if not any(_mdiff(t_start, m) < -PRETREND_LEADS for m in present_treated):
        return None

    dists = sorted({d for d, _, _ in raw})
    n_m = len(months)
    d_index = {d: j for j, d in enumerate(dists)}
    rel_index = {r: k for k, r in enumerate(rels_present)}
    k_total = n_m + (len(dists) - 1) + len(rels_present)
    rows: list[list[int]] = []
    ys: list[float] = []
    for d, m, y in raw:
        ix = [idx_of[m]]
        j = d_index[d]
        if j > 0:
            ix.append(n_m + j - 1)
        if d == treated:
            r = _mdiff(t_start, m)
            if r >= -PRETREND_LEADS:
                ix.append(n_m + len(dists) - 1 + rel_index[r])
        rows.append(ix)
        ys.append(y)

    # Sparse normal equations: every regressor is a 0/1 dummy.
    A = [[0.0] * k_total for _ in range(k_total)]
    bv = [0.0] * k_total
    for ix, y in zip(rows, ys):
        for i in ix:
            bv[i] += y
            for j in ix:
                A[i][j] += 1.0
    beta = [row[0] for row in _mat_solve(A, [[v] for v in bv])]
    resid_ss = 0.0
    for ix, y in zip(rows, ys):
        e = y - sum(beta[i] for i in ix)
        resid_ss += e * e
    dof = max(1, len(ys) - k_total)
    sigma2 = resid_ss / dof
    Ainv = _mat_inv(A)

    def coef(r: int) -> float:
        return beta[n_m + len(dists) - 1 + rel_index[r]]

    def se(r: int) -> float:
        i = n_m + len(dists) - 1 + rel_index[r]
        return math.sqrt(max(1e-12, sigma2 * Ainv[i][i]))

    window_rels = [r for r in rels_present if 0 <= r < L]
    eff = sum(coef(r) for r in window_rels) / len(window_rels)
    prestat = (sum((coef(r) / se(r)) ** 2 for r in leads_present)
               / len(leads_present))
    return {
        "beta": eff,
        "prestat": prestat,
        "n_leads": len(leads_present),
        "path": [{"month": add_months(t_start, r), "rel": r,
                  "coef": coef(r), "se": se(r)} for r in rels_present],
    }


def _stack_placebos(counts: dict, cell: dict,
                    exclude_3d: bool) -> list[tuple[str, date]]:
    """In-space x in-time placebo assignments for one stack: every
    (district, window start) whose same-length window fits the span,
    is unmasked for that district, has 12 prior span months, and (for the
    treated district itself) does not overlap the documented window."""
    months = _stack_span(cell)
    L = len(_estimable_window(cell))
    actual = set(_estimable_window(cell))
    out = []
    for d in DISTRICTS:
        for i in range(MIN_PLACEBO_PRE_MONTHS, len(months) - L + 1):
            t = months[i]
            win = month_range(t, add_months(t, L))
            if d == cell["district"]:
                if t == cell["start"]:
                    continue                      # the real assignment
                if any(m in actual for m in win):
                    continue                      # overlaps the real window
            if any(_is_masked(d, m, exclude_3d) for m in win):
                continue
            out.append((d, t))
    return out


def _pooled_candidates(cell: dict, exclude_3d: bool) -> list[str]:
    """In-space candidates for the pooled permutation: districts whose
    documented-window months are fully unmasked (the treated district's own
    real assignment is excluded)."""
    win = _estimable_window(cell)
    out = []
    for d in DISTRICTS:
        if d == cell["district"]:
            continue
        if all(not _is_masked(d, m, exclude_3d) for m in win):
            out.append(d)
    return out


def _pooled_beta(counts: dict, cells: Sequence[dict],
                 assignment: dict[str, str], exclude_3d: bool
                 ) -> Optional[dict]:
    """Pooled stacked DiD coefficient under an assignment mapping cell key ->
    treated district (the real one or a placebo). Stack-by-district and
    stack-by-month effects are absorbed by alternating projections; beta is
    the FWL coefficient of the treated-window dummy. Also returns per-stack
    contribution weights (share of residualized treatment variance)."""
    ys: list[float] = []
    ts: list[float] = []
    g_dist: list[int] = []
    g_month: list[int] = []
    stack_of: list[int] = []
    g1_key: dict[tuple[int, str], int] = {}
    g2_key: dict[tuple[int, date], int] = {}
    for s_i, cell in enumerate(cells):
        treated = assignment[cell["key"]]
        months = _stack_span(cell)
        win = set(_estimable_window(cell))
        for d in DISTRICTS:
            dm = counts.get(d, {})
            for m in months:
                n = dm.get(m, 0)
                if n <= 0:
                    continue
                in_win = d == treated and m in win
                if _is_masked(d, m, exclude_3d) and not in_win:
                    continue
                ys.append(math.log(float(n)))
                ts.append(1.0 if in_win else 0.0)
                g_dist.append(g1_key.setdefault((s_i, d), len(g1_key)))
                g_month.append(g2_key.setdefault((s_i, m), len(g2_key)))
                stack_of.append(s_i)
    if not any(t > 0 for t in ts):
        return None

    def demean(v: list[float]) -> list[float]:
        v = list(v)
        for _ in range(60):
            shift = 0.0
            for groups, size in ((g_dist, len(g1_key)), (g_month, len(g2_key))):
                tot = [0.0] * size
                cnt = [0] * size
                for i, g in enumerate(groups):
                    tot[g] += v[i]
                    cnt[g] += 1
                for i, g in enumerate(groups):
                    mu = tot[g] / cnt[g]
                    v[i] -= mu
                    shift = max(shift, abs(mu))
            if shift < 1e-11:
                break
        return v

    yr, tr = demean(ys), demean(ts)
    denom = sum(x * x for x in tr)
    if denom <= 0:
        return None
    beta = sum(a * b for a, b in zip(tr, yr)) / denom
    weights: dict[str, float] = {}
    betas: dict[str, float] = {}
    for s_i, cell in enumerate(cells):
        d_s = sum(tr[i] * tr[i] for i in range(len(tr)) if stack_of[i] == s_i)
        n_s = sum(tr[i] * yr[i] for i in range(len(tr)) if stack_of[i] == s_i)
        weights[cell["key"]] = d_s / denom
        betas[cell["key"]] = n_s / d_s if d_s > 0 else 0.0
    return {"beta": beta, "weights": weights, "stack_betas": betas}


def _perm_p(actual: float, placebo: Sequence[float], tail: str) -> float:
    if tail == "low":
        worse = sum(1 for x in placebo if x <= actual)
    else:
        worse = sum(1 for x in placebo if x >= actual)
    return (1 + worse) / (1 + len(placebo))


def _pooled_permutation(counts: dict, cells: Sequence[dict],
                        exclude_3d: bool) -> Optional[dict]:
    actual_assign = {c["key"]: c["district"] for c in cells}
    actual = _pooled_beta(counts, cells, actual_assign, exclude_3d)
    if actual is None:
        return None
    cand = [_pooled_candidates(c, exclude_3d) for c in cells]
    placebo_betas: list[float] = []
    assigns: list[dict[str, str]] = [{}]
    for c, options in zip(cells, cand):
        assigns = [dict(a, **{c["key"]: d}) for a in assigns for d in options]
    for a in assigns:
        r = _pooled_beta(counts, cells, a, exclude_3d)
        if r is not None:
            placebo_betas.append(r["beta"])
    p = _perm_p(actual["beta"], placebo_betas, "low")
    return {"beta": actual["beta"], "p_perm": p,
            "n_placebo": len(placebo_betas),
            "discovery": actual["beta"] < 0 and p <= PERMUTATION_ALPHA,
            "weights": actual["weights"],
            "stack_betas": actual["stack_betas"]}


def theft_exit_from_counts(counts: dict[str, dict[date, int]]) -> dict:
    """Full analysis from per-district monthly outcome counts. Pure, so the
    whole pipeline is testable on synthetic data."""
    cells_out: list[dict] = []
    for cell in CELLS:
        fit = _fit_stack(counts, cell, cell["district"], cell["start"], True)
        if fit is None:
            raise RuntimeError(f"stack {cell['key']} could not be fit; "
                               "archive is missing window months")
        placebos = _stack_placebos(counts, cell, True)
        pl_beta: list[float] = []
        pl_prestat: list[float] = []
        for d, t in placebos:
            try:
                r = _fit_stack(counts, cell, d, t, True)
            except ValueError:      # singular placebo design; skip, count not
                continue
            if r is not None:
                pl_beta.append(r["beta"])
                pl_prestat.append(r["prestat"])
        p_cell = _perm_p(fit["beta"], pl_beta, "low")
        p_pre = _perm_p(fit["prestat"], pl_prestat, "high")
        # Detection floor from the placebo distribution itself: the deficit
        # size a placebo produces 5% of the time (design-based MDE).
        neg = sorted(pl_beta)
        q05 = neg[int(math.floor(0.05 * len(neg)))] if neg else 0.0
        mde_pct = max(0.0, (1.0 - math.exp(q05)) * 100.0)
        win = _estimable_window(cell)
        base = (sum(counts.get(cell["district"], {}).get(m, 0) for m in win)
                / max(1, len(win)))
        doc = cell["doc_per_month"]
        expected = doc / (doc + base) * 100.0 if base else 0.0
        cells_out.append({
            "key": cell["key"], "district": cell["district"],
            "label": cell["label"], "docs": cell["docs"],
            "beta": fit["beta"], "se_nw": 0.0,
            "p_perm": p_cell, "p_adjusted": 1.0, "discovery": False,
            "n_placebo": len(pl_beta),
            "pretrend": {"joint_p": p_pre, "ok": p_pre > PRETREND_ALPHA,
                         "stat": fit["prestat"]},
            "bridge": {"doc_per_month": doc, "base_per_month": base,
                       "expected_dip_pct": expected, "mde_pct": mde_pct,
                       "below_floor": expected < mde_pct},
            "event_path": fit["path"],
        })

    bh = benjamini_hochberg([c["p_perm"] for c in cells_out], BH_Q)
    for c, adj, disc in zip(cells_out, bh["adjusted"], bh["discoveries"]):
        c["p_adjusted"] = adj
        c["discovery"] = bool(disc) and c["beta"] < 0 and c["pretrend"]["ok"]

    active = [cc for cc, c in zip(CELLS, cells_out) if c["pretrend"]["ok"]]
    pooled = (_pooled_permutation(counts, active, True) if active else None)
    sens = (_pooled_permutation(counts, active, False) if active else None)
    if pooled is None:
        pooled = {"beta": 0.0, "p_perm": 1.0, "n_placebo": 0,
                  "discovery": False, "weights": {}, "stack_betas": {}}
    if sens is None:
        sens = {"beta": 0.0, "p_perm": 1.0, "discovery": False}

    # Descriptive context for the page's exhibits (no inference): yearly
    # outcome sums per district, so the raw divergence behind a pre-trend
    # verdict can be shown in plain counts rather than described. COMPLETE
    # calendar years only: the estimation span starts mid-year and ends at
    # the surge boundary, and a partial year quoted as a year would inflate
    # every ratio built from it.
    yearly: dict[str, dict[int, int]] = {}
    for d in DISTRICTS:
        by_year: dict[int, list[int]] = {}
        for m, n in counts.get(d, {}).items():
            by_year.setdefault(m.year, []).append(n)
        yearly[d] = {y: sum(v) for y, v in by_year.items() if len(v) == 12}

    result = {
        "pooled": pooled,
        "cells": cells_out,
        "sensitivity_3d": {"beta": sens["beta"], "p_perm": sens["p_perm"],
                           "discovery": sens["discovery"]},
        "backfill": {"recovered_rows": BACKFILL_RECOVERED_ROWS},
        "context_yearly": yearly,
    }

    # Pre-registered THRIVE timing call for the anchor (rule fixed above).
    anchor = cell_by_key(result, ANCHOR_KEY)
    if anchor is not None:
        pre = [p["coef"] for p in anchor["event_path"]
               if p["month"] in ANCHOR_PRE_THRIVE_MONTHS]
        pre_mean = sum(pre) / len(pre) if pre else 0.0
        anchor["thrive_timing"] = {
            "pre_thrive_mean": pre_mean,
            "window_mean": anchor["beta"],
            "opens_before": bool(
                anchor["discovery"] and anchor["beta"] < 0
                and pre_mean <= THRIVE_OPENS_BEFORE_FRACTION * anchor["beta"]),
        }
    return result


_MONTHLY_DISTRICT_SQL = """
SELECT district, date_trunc('month', report_dat AT TIME ZONE 'UTC')::date AS month,
       COUNT(*)::int AS n
FROM crimes
WHERE city='dc' AND offense = ANY($1::text[])
  AND district = ANY($2::text[])
  AND CAST(report_dat AT TIME ZONE 'UTC' AS date) >= $3
  AND CAST(report_dat AT TIME ZONE 'UTC' AS date) < $4
GROUP BY 1, 2
"""


async def theft_exit_did(conn) -> dict:
    """Run the pre-registered analysis against the incident archive.

    The fetch starts at January of the earliest stack's first year rather
    than at the stack boundary itself. Estimation is unaffected (every fit
    iterates its own stack span), but it lets the page's descriptive
    context quote whole calendar years instead of part-years.
    """
    lo = min(_stack_span(c)[0] for c in CELLS)
    rows = await conn.fetch(_MONTHLY_DISTRICT_SQL, list(OUTCOME_OFFENSES),
                            list(DISTRICTS), date(lo.year, 1, 1), SURGE_START)
    if not rows:
        raise RuntimeError("no incident archive - run the crime ingest first.")
    counts: dict[str, dict[date, int]] = {}
    for r in rows:
        counts.setdefault(r["district"], {})[r["month"]] = r["n"]
    return theft_exit_from_counts(counts)
