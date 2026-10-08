"""Classification-mix changepoint analysis: the data-reliability audit.

The July 2026 DC OIG inspection (OIG No. 26-E-03-FA0) found that MPD's
internal controls over crime classification deteriorated for more than a
decade: the Staff Review Unit, the independent function that reviewed crime
reports before they became official statistics, was disbanded around 2013
during the transition to a new records system and never replaced. The OIG
found weak controls, NOT intentional manipulation, and reviewed 2015-2025
data. This module asks the one question our archive can ask about that
finding: did the loss of independent review leave a detectable fingerprint in
the offense-classification mix, 2008 to present?

WHAT THIS CAN AND CANNOT SEE. The public incident feed carries only the nine
DC index-crime categories. A reclassification BETWEEN two visible categories
(ADW -> robbery, theft -> theft from auto) shows up as substitution in the
shares. A downgrade OUT of the index set (ADW -> simple assault) removes the
incident from this dataset entirely and is invisible here. So the analysis
detects shifts between visible categories and unexplained level changes
within one; it cannot directly observe exits from the feed.

Design, built to survive a hostile statistician's first pass:
  * Monthly category shares 2008 to the last complete month, citywide.
  * Detected breaks: binary segmentation on deseasonalized shares
    (mean-shift cost, BIC-style penalty, 12-month minimum segments).
  * A-priori 2013 test: interrupted time-series segmented regression (level
    shift AND slope change at Jan 2013, calendar-month dummies) on a
    symmetric five-year window each side of the break, with NEWEY-WEST
    standard errors, because monthly shares are autocorrelated and naive OLS
    intervals would overstate significance. The window (2008-2017) excludes
    the COVID discontinuity and the 2023 motor vehicle theft wave from the
    test by construction; the full-window scan below still sees them.
  * Multiple testing: Benjamini-Hochberg across the whole ITS family
    (9 category series + 3 substitution pairs = 12 tests); uncorrected
    per-test results are reported but discoveries are the corrected ones.
  * Compositional guard: the nine shares sum to 1, so a real surge in one
    category mechanically depresses every other share. Raw counts are
    returned alongside shares, and the three substitution pairs use
    WITHIN-PAIR shares (A/(A+B)), which the compositional coupling cannot
    move.
  * Known confounds carried as first-class data: the COVID-19 discontinuity
    (never attributable to classification practice), the Mark43 RMS
    migration, and the thin post-training window after Feb 2026.

Same contracts as every other engine: pure functions for the math,
RuntimeError when data is missing so surfaces degrade instead of erroring.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Optional, Sequence

from database.crime_baseline import VICTIM_OFFENSES, VIOLENT_OFFENSES

ARCHIVE_START = date(2008, 1, 1)

# Reference events, each labeled with WHAT HAPPENED, not just a year.
# The SRU date and the Cobalt-era RMS transition come from the OIG report
# (disbanded "around 2013"); the Mark43 go-live is dated by the vendor
# (August 2015), since the OIG documents the transition but not its date.
EVENTS: tuple[dict[str, Any], ...] = (
    {"key": "sru", "month": date(2013, 1, 1),
     "short": "Review unit disbanded (2013)",
     "label": ("2013: MPD disbands the Staff Review Unit, the independent "
               "function that reviewed crime reports before they became "
               "official statistics (per the OIG; no replacement until 2026)")},
    {"key": "mark43", "month": date(2015, 8, 1),
     "short": "Mark43 RMS goes live (Aug 2015)",
     "label": ("Aug 2015: the Mark43 records management system goes live at "
               "MPD (migration documented by the OIG; go-live date per the "
               "vendor). System migrations can shift recorded classifications "
               "on their own")},
    {"key": "smith", "month": date(2023, 8, 1),
     "short": "Pressure era begins (Aug 2023)",
     "label": ("Aug 2023: Pamela Smith becomes chief; the House Oversight and "
               "internal affairs reports date the documented pressure on "
               "crime classifications to her tenure")},
    {"key": "exposure", "month": date(2025, 12, 1),
     "short": "Scandal breaks (Dec 2025)",
     "label": ("Dec 2025: the House Oversight and DOJ reports document "
               "crime-data manipulation and the chief resigns")},
    {"key": "training", "month": date(2026, 2, 1),
     "short": "Classification training (Feb 2026)",
     "label": ("Feb 2026: MPD launches mandatory DC Code classification "
               "training, the first of the post-OIG reforms; post-reform "
               "data is still thin")},
)
COVID_BAND = (date(2020, 3, 1), date(2021, 6, 1))
COVID_LABEL = ("COVID-19 disruption: reported crime shifted abruptly for "
               "reasons unrelated to classification practice")

BREAK_MONTH = date(2013, 1, 1)     # the a-priori ITS break point
# The ITS is estimated on a symmetric window around the break (five years each
# side, so Jan 2008 - Dec 2017). Two reasons, both about surviving hostile
# review: (1) the standard segmented-regression spec (level + slope change)
# is only credible locally; over the full 18-year window any long-run
# curvature in a series loads onto the break dummy and manufactures "shifts";
# (2) the window ends before the 2020 COVID discontinuity and the 2023 motor
# vehicle theft wave, so neither confound can touch the a-priori test by
# construction. The changepoint scan still covers the full archive.
ITS_WINDOW_MONTHS = 60
# Placebo break dates for the specificity check: the same family test run at
# boundaries where nothing control-related happened. Chosen so every placebo
# window stays clear of the COVID band (break + 60 months <= Mar 2020) and
# a-priori (calendar Januaries around the real break, none cherry-picked).
# If placebo dates fire as often as 2013, the mix is churning generally and
# the 2013 boundary cannot be singled out; the page must say which it is.
PLACEBO_BREAKS = (date(2011, 1, 1), date(2012, 1, 1),
                  date(2014, 1, 1), date(2015, 1, 1))
MIN_SEGMENT_MONTHS = 12

# ── The documented manipulation era (2023-2025) ──────────────────────────────
# Boundaries verified from reporting (see the page's context block for links):
# Pamela Smith became chief in July 2023 (first full month August); the House
# Oversight interim report and DOJ findings landed in December 2025 and the
# chief resigned the same month; the 554-page MPD internal affairs report
# surfaced in May 2026. The federal law enforcement surge began August 2025
# and changed real enforcement and possibly real crime: like COVID at 2013,
# it is a boundary this analysis cannot attribute past.
ERA_BREAK = date(2023, 8, 1)
ERA_WINDOW_MONTHS = 24        # +-24: avoids the COVID band AND ends pre-surge
SURGE_START = date(2025, 8, 1)
SURGE_LABEL = ("Federal law enforcement surge begins Aug 2025: real "
               "enforcement, and possibly real crime, changed; nothing after "
               "this boundary is attributed to classification practice")
EXPOSURE_MONTH = date(2025, 12, 1)
REVERSION_START = date(2021, 8, 1)
# Placebos for the era test. A +-24 window that avoids both COVID and the
# surge fits only the real boundary, so placebos come from the clean
# pre-COVID era with the same window shape; that limitation is stated on
# the page.
ERA_PLACEBO_BREAKS = (date(2011, 8, 1), date(2013, 8, 1),
                      date(2015, 8, 1), date(2017, 8, 1))

# Directional predictions fixed a-priori from the internal affairs report's
# documented downgrade paths (robberies logged as theft; weapons assaults
# reduced to lesser offenses; shootings relabeled "injured person", which
# exits the index feed entirely). Sign -1 = the share falls under pressure.
DIRECTIONAL_CONTRASTS: tuple[dict[str, Any], ...] = (
    {"key": "adw_violent", "sign": -1,
     "path": "weapons assaults downgraded below ADW (many exit the feed)"},
    {"key": "robbery_theft", "sign": -1,
     "path": "robberies logged as theft"},
    {"key": "theft_split", "sign": -1,
     "path": "thefts shifted toward the lesser theft category"},
    {"key": "ASSAULT W/DANGEROUS WEAPON", "sign": -1,
     "path": "ADW share of the whole mix falls as downgrades exit"},
)

# Countable scale from the PRIMARY document: MPD Internal Affairs Division,
# Final Investigative Report IS #26-000051 (March 5, 2026; 554 pages;
# published by WJLA May 2026). The report states NO comprehensive total and
# says its fullest tally (the 7DDU spreadsheet) is "by no means a complete
# record" (p.526), so the total below is OUR COMPUTED SUM of the report's
# per-official audit counts, each page-cited, and remains a floor:
#   Cmdr M. Pulliam 157 reclassified in total (p.16); Capt Donigian 363
#   edited (p.318); Capt Haskis ~194 edited (p.334); Capt Rivers 131
#   reclassified (p.217); Capt R. Pulliam 47 modified (p.370); Capt Merrill
#   ~30 reclassified (p.533). Sum = 922. Not all altered the public DCR
#   (e.g. 30 of Pulliam's stayed in-index as Shoplifting, p.16-17), and
#   several more officials have no published count.
DOCUMENTED_EDITED_REPORTS = 922
DOCUMENTED_OFFICIAL_COUNTS = (
    ("Cmdr M. Pulliam", 157, "p. 16"),
    ("Capt Donigian", 363, "p. 318"),
    ("Capt Haskis", 194, "p. 334"),
    ("Capt Rivers", 131, "p. 217"),
    ("Capt R. Pulliam", 47, "p. 370"),
    ("Capt Merrill", 30, "p. 533"),
)
# Per-downgrade-path documented floors, from the report's own counts. These
# make the detectability comparison per contrast far sharper than a pooled
# figure: the documented manipulation is overwhelmingly THEFT reports moved
# OUT of the index feed; documented robbery and ADW reclassifications are
# single- and low-double-digit.
#   robbery: Pulliam 3 robberies -> non-reportable (p.16) plus at least 2
#     documented robbery -> simple assault changes by Haskis (pp.256, 309).
#   adw: Pulliam 11 ADWs -> non-reportable (p.16); the 7DDU spreadsheet's
#     "large number" of misclassified ADWs carries no count (p.21).
#   theft_exit (thefts moved out of the index feed entirely): Pulliam 83
#     (45 theft-related -> incidents + 38 DCR -> non-reportable, p.16),
#     Haskis ~151 Theft -> TPWR (p.335), Rivers 129 theft-forms -> Lost
#     Property/TPWR/Misc/etc (p.217), R. Pulliam 21 Theft -> TPWR (p.370),
#     Merrill ~22 (13 Miscellaneous + 9 Attempt to Locate, p.533) = 406
#     explicitly counted; most of Donigian's 363 sit on top uncounted
#     ("primarily" Theft -> TPWR/Lost Property, p.318).
DOCUMENTED_PATH_FLOORS = {"robbery": 5, "adw": 11, "theft_exit": 406}
IA_REPORT_CITE = ("MPD Internal Affairs Division, Final Investigative Report "
                  "IS #26-000051, March 5, 2026")
BINSEG_PENALTY_MULT = 3.0          # x sigma^2 log(n); slightly conservative
BH_Q = 0.05

# The three pairs where reclassification would show as substitution. Each is
# tested on its WITHIN-PAIR share A/(A+B): immune to the compositional
# coupling that lets a surge in a third category move every global share.
PAIRS: tuple[dict[str, Any], ...] = (
    {"key": "theft_split", "a": "THEFT F/AUTO", "b": ("THEFT/OTHER",),
     "label": "Theft from auto, as a share of all theft"},
    {"key": "robbery_theft", "a": "ROBBERY", "b": ("THEFT/OTHER",),
     "label": "Robbery, as a share of robbery + theft"},
    {"key": "adw_violent", "a": "ASSAULT W/DANGEROUS WEAPON",
     "b": tuple(o for o in VIOLENT_OFFENSES if o != "ASSAULT W/DANGEROUS WEAPON"),
     "label": "ADW, as a share of all violent categories"},
)

CLASSIFICATION_DEFINITION = (
    "Each series is a monthly share of the citywide classification mix, "
    "2008 to the last complete month. Detected breaks come from binary "
    "segmentation on the deseasonalized series over the full window; the "
    "a-priori 2013 test is a segmented interrupted time-series regression "
    "(level shift and slope change at Jan 2013, calendar-month dummies) on "
    "the symmetric 2008-2017 window, with Newey-West standard errors, "
    "corrected for multiple testing across the full family."
)


# ── Calendar helpers (same conventions as database/monthly.py) ────────────────
def month_start(value: date) -> date:
    return value.replace(day=1)


def add_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def month_range(start: date, end_exclusive: date) -> list[date]:
    out = []
    cursor = month_start(start)
    while cursor < end_exclusive:
        out.append(cursor)
        cursor = add_months(cursor, 1)
    return out


def month_index(months: Sequence[date], target: date) -> Optional[int]:
    """Index of the first month >= target, None if past the window."""
    for i, m in enumerate(months):
        if m >= target:
            return i
    return None


# ── Pure math: shares ─────────────────────────────────────────────────────────
def share_matrix(rows: list[dict], start: date, end_exclusive: date) -> dict:
    """(month, offense, n) rows -> zero-filled monthly counts, totals, and
    shares per offense category. Months with zero incidents overall get None
    shares (never a fabricated 0)."""
    months = month_range(start, end_exclusive)
    idx = {m: i for i, m in enumerate(months)}
    counts = {off: [0] * len(months) for off in VICTIM_OFFENSES}
    for r in rows:
        m = month_start(r["month"])
        if r["offense"] in counts and m in idx:
            counts[r["offense"]][idx[m]] += int(r["n"])
    totals = [sum(counts[off][i] for off in VICTIM_OFFENSES)
              for i in range(len(months))]
    shares = {
        off: [(counts[off][i] / totals[i]) if totals[i] else None
              for i in range(len(months))]
        for off in VICTIM_OFFENSES
    }
    return {"months": months, "counts": counts, "totals": totals, "shares": shares}


def pair_share(counts: dict[str, list[int]], a: str, b: Sequence[str]) -> list[Optional[float]]:
    """Within-pair share A/(A+B): the substitution statistic the global
    compositional coupling cannot move."""
    n = len(counts[a])
    out: list[Optional[float]] = []
    for i in range(n):
        na = counts[a][i]
        nb = sum(counts[x][i] for x in b)
        out.append(na / (na + nb) if (na + nb) else None)
    return out


def deseasonalize(y: Sequence[float], months: Sequence[date]) -> list[float]:
    """Remove calendar-month means (computed over the full window), keeping
    the grand mean so the series stays in level units."""
    by_cal: dict[int, list[float]] = {}
    for v, m in zip(y, months):
        by_cal.setdefault(m.month, []).append(v)
    cal_mean = {k: sum(vs) / len(vs) for k, vs in by_cal.items()}
    grand = sum(y) / len(y)
    return [v - cal_mean[m.month] + grand for v, m in zip(y, months)]


# ── Pure math: binary segmentation ────────────────────────────────────────────
def _sse(prefix: list[float], prefix_sq: list[float], a: int, b: int) -> float:
    """Sum of squared deviations from the mean on y[a:b] via prefix sums."""
    n = b - a
    if n <= 0:
        return 0.0
    s = prefix[b] - prefix[a]
    q = prefix_sq[b] - prefix_sq[a]
    return max(0.0, q - s * s / n)


def binary_segmentation(y: Sequence[float], min_seg: int = MIN_SEGMENT_MONTHS,
                        penalty: Optional[float] = None) -> list[int]:
    """Mean-shift changepoints by binary segmentation.

    A split is accepted only when it reduces the segment SSE by more than the
    penalty (default BINSEG_PENALTY_MULT x sigma^2 log n, with sigma^2
    estimated from first differences, which is robust to the level shifts
    being tested for). Returns sorted indices i where a new segment starts at
    y[i]. Deliberately conservative: this feeds a page that must not
    manufacture breaks.
    """
    n = len(y)
    if n < 2 * min_seg:
        return []
    if penalty is None:
        diffs = [y[i + 1] - y[i] for i in range(n - 1)]
        dm = sum(diffs) / len(diffs)
        var_diff = sum((d - dm) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        sigma2 = var_diff / 2.0
        if sigma2 <= 0:
            return []
        penalty = BINSEG_PENALTY_MULT * sigma2 * math.log(n)

    prefix = [0.0]
    prefix_sq = [0.0]
    for v in y:
        prefix.append(prefix[-1] + v)
        prefix_sq.append(prefix_sq[-1] + v * v)

    breaks: list[int] = []

    def split(a: int, b: int) -> None:
        if b - a < 2 * min_seg:
            return
        whole = _sse(prefix, prefix_sq, a, b)
        best_k: Optional[int] = None
        best_cost = math.inf
        for k in range(a + min_seg, b - min_seg + 1):
            cost = _sse(prefix, prefix_sq, a, k) + _sse(prefix, prefix_sq, k, b)
            if cost < best_cost:
                best_k, best_cost = k, cost
        if best_k is not None and whole - best_cost > penalty:
            breaks.append(best_k)
            split(a, best_k)
            split(best_k, b)

    split(0, n)
    return sorted(breaks)


# ── Pure math: OLS with Newey-West standard errors ────────────────────────────
def _mat_solve(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Solve A X = B by Gauss-Jordan with partial pivoting (A is k x k)."""
    k = len(A)
    aug = [list(A[i]) + list(B[i]) for i in range(k)]
    w = len(aug[0])
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular design matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        d = aug[col][col]
        for j in range(w):
            aug[col][j] /= d
        for r in range(k):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                for j in range(w):
                    aug[r][j] -= f * aug[col][j]
    return [row[k:] for row in aug]


def _mat_inv(A: list[list[float]]) -> list[list[float]]:
    k = len(A)
    return _mat_solve(A, [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)])


def _xtx(X: list[list[float]]) -> list[list[float]]:
    k = len(X[0])
    out = [[0.0] * k for _ in range(k)]
    for row in X:
        for i in range(k):
            xi = row[i]
            if xi:
                for j in range(k):
                    out[i][j] += xi * row[j]
    return out


def normal_p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def newey_west_lags(n: int) -> int:
    """The standard plug-in truncation: floor(4 (n/100)^(2/9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _ols_nw(X: list[list[float]], yv: list[float], idx: int,
            nw_lags: int) -> dict:
    """OLS with Newey-West (Bartlett) standard errors on coefficient idx.
    The shared core behind every regression on this page."""
    n, k = len(X), len(X[0])
    xtx = _xtx(X)
    xty = [[sum(X[t][i] * yv[t] for t in range(n))] for i in range(k)]
    beta = [b[0] for b in _mat_solve(xtx, xty)]
    resid = [yv[t] - sum(X[t][i] * beta[i] for i in range(k)) for t in range(n)]
    xtx_inv = _mat_inv(xtx)

    # Meat of the sandwich: S = sum_t e_t^2 x_t x_t'
    #                         + sum_l w_l sum_t (x_t e_t e_{t-l} x_{t-l}' + sym)
    S = [[0.0] * k for _ in range(k)]
    for t in range(n):
        e2 = resid[t] * resid[t]
        for i in range(k):
            xi = X[t][i]
            if xi:
                for j in range(k):
                    S[i][j] += e2 * xi * X[t][j]
    for lag in range(1, nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        for t in range(lag, n):
            ee = resid[t] * resid[t - lag]
            for i in range(k):
                for j in range(k):
                    S[i][j] += w * ee * (X[t][i] * X[t - lag][j]
                                         + X[t - lag][i] * X[t][j])

    # Var(beta) = (X'X)^-1 S (X'X)^-1, small-sample factor n/(n-k).
    tmp = [[sum(xtx_inv[i][a] * S[a][j] for a in range(k)) for j in range(k)]
           for i in range(k)]
    cov_nw = [[sum(tmp[i][a] * xtx_inv[a][j] for a in range(k)) * n / (n - k)
               for j in range(k)] for i in range(k)]

    sse = sum(e * e for e in resid)
    sigma2_ols = sse / (n - k)
    return {
        "beta": beta[idx],
        "se_nw": math.sqrt(max(0.0, cov_nw[idx][idx])),
        "se_ols": math.sqrt(max(0.0, sigma2_ols * xtx_inv[idx][idx])),
        "n": n,
    }


def its_level_shift(y: Sequence[Optional[float]], months: Sequence[date],
                    break_month: date = BREAK_MONTH,
                    nw_lags: Optional[int] = None,
                    window_months: int = ITS_WINDOW_MONTHS) -> Optional[dict]:
    """Interrupted time-series estimate of the level shift at break_month.

    Segmented regression on a symmetric window (window_months each side of
    the break): intercept, linear trend, post-break indicator, post-break
    slope change, and 11 calendar-month dummies. The trend is centered at
    the break so the post coefficient is the jump AT the boundary. Newey-West
    (Bartlett kernel) standard errors on that coefficient, because monthly
    shares are autocorrelated and naive OLS intervals are anti-conservative.
    The naive OLS p-value is also returned so the correction is auditable,
    but p_nw is the number that counts.
    """
    lo = add_months(break_month, -window_months)
    hi = add_months(break_month, window_months)
    pts = [(v, m) for v, m in zip(y, months)
           if v is not None and lo <= m < hi]
    min_side = max(MIN_SEGMENT_MONTHS, window_months // 2)
    if len(pts) < 2 * min_side:
        return None
    n = len(pts)
    bi = month_index([m for _, m in pts], break_month)
    if bi is None or bi < min_side or n - bi < min_side:
        return None
    if nw_lags is None:
        nw_lags = newey_west_lags(n)

    # Columns: [intercept, u, post, post*u, Feb..Dec dummies], u = years since
    # the break (negative before it). post is column 2: the jump at u = 0.
    X = []
    for t, (_, m) in enumerate(pts):
        u = (t - bi) / 12.0
        post = 1.0 if t >= bi else 0.0
        row = [1.0, u, post, post * u]
        row += [1.0 if m.month == cal else 0.0 for cal in range(2, 13)]
        X.append(row)
    yv = [v for v, _ in pts]

    fit = _ols_nw(X, yv, 2, nw_lags)
    shift, se_nw, se_ols = fit["beta"], fit["se_nw"], fit["se_ols"]
    return {
        "break_month": break_month,
        "window": (pts[0][1], pts[-1][1]),
        "n": n,
        "nw_lags": nw_lags,
        "shift": shift,
        "se_nw": se_nw,
        "p_nw": normal_p_two_sided(shift / se_nw) if se_nw else 1.0,
        "ci_low": shift - 1.96 * se_nw,
        "ci_high": shift + 1.96 * se_nw,
        "se_ols": se_ols,
        "p_ols": normal_p_two_sided(shift / se_ols) if se_ols else 1.0,
    }


def one_sided_p(shift: float, se: float, sign: int) -> float:
    """One-sided p-value for an a-priori directional prediction (sign +1/-1)."""
    if se <= 0:
        return 1.0
    z = shift / se
    p_two = normal_p_two_sided(z)
    return p_two / 2.0 if z * sign > 0 else 1.0 - p_two / 2.0


def era_reversion(y: Sequence[Optional[float]], months: Sequence[date],
                  era_sign: int, start: date = REVERSION_START) -> Optional[dict]:
    """The decline-then-reversion signature test for one directional series.

    One regression over start..end with three step dummies (pressure era from
    Aug 2023, federal surge from Aug 2025, post-exposure from Dec 2025), a
    linear trend, and calendar-month dummies, Newey-West errors. The
    coefficient of interest is the post-exposure step: reversion predicts it
    moves OPPOSITE to the era prediction (era_sign), one-sided. Two caveats
    travel with every use: post-exposure months are few, and every one of
    them sits past the surge boundary, so reversion cannot be cleanly
    attributed to exposure versus the surge.
    """
    pts = [(v, m) for v, m in zip(y, months) if v is not None and m >= start]
    n = len(pts)
    post_expo = sum(1 for _, m in pts if m >= EXPOSURE_MONTH)
    if n < 3 * MIN_SEGMENT_MONTHS or post_expo < 3:
        return None
    X = []
    for t, (_, m) in enumerate(pts):
        row = [1.0, t / 12.0,
               1.0 if m >= ERA_BREAK else 0.0,
               1.0 if m >= SURGE_START else 0.0,
               1.0 if m >= EXPOSURE_MONTH else 0.0]
        row += [1.0 if m.month == cal else 0.0 for cal in range(2, 13)]
        X.append(row)
    yv = [v for v, _ in pts]
    nw_lags = newey_west_lags(n)
    fit = _ols_nw(X, yv, 4, nw_lags)
    rev_sign = -era_sign
    return {
        "n": n,
        "post_exposure_months": post_expo,
        "nw_lags": nw_lags,
        "shift": fit["beta"],
        "se_nw": fit["se_nw"],
        "rev_sign": rev_sign,
        "p_one": one_sided_p(fit["beta"], fit["se_nw"], rev_sign),
    }


def magnitude_bridge(matrix: dict, months: Sequence[date],
                     directional: Sequence[dict]) -> Optional[dict]:
    """Order-of-magnitude bridge from the internal affairs report's own
    numbers (per-official audits, page-cited in the constants above) to what
    this test can detect.

    Two layers, both compared against each directional test's minimum
    detectable effect (1.645 x its Newey-West SE, one-sided 5%):
      * documented per-path scenarios: the report's counted robbery, ADW,
        and theft-exit reclassifications spread over the pressure-era months;
      * a stress ceiling: the ENTIRE computed sum of edited reports (922)
        concentrated on a single path, the most visible the documented
        manipulation could possibly have been.
    If the documented scale sits below detectability, a null is the EXPECTED
    result at that scale, and the page says so.
    """
    idxs = [i for i, m in enumerate(months) if ERA_BREAK <= m < SURGE_START]
    if not idxs:
        return None
    span = max(1, (EXPOSURE_MONTH.year - ERA_BREAK.year) * 12
               + EXPOSURE_MONTH.month - ERA_BREAK.month)          # 28 months

    def mean(off: str) -> float:
        return sum(matrix["counts"][off][i] for i in idxs) / len(idxs)

    rob, theft = mean("ROBBERY"), mean("THEFT/OTHER")
    adw = mean("ASSAULT W/DANGEROUS WEAPON")
    violent = adw + rob + mean("SEX ABUSE") + mean("HOMICIDE")
    total = sum(matrix["totals"][i] for i in idxs) / len(idxs)

    def robbery_pp(n_reports: float) -> float:
        return (n_reports / span) / (rob + theft) * 100.0

    def adw_pp(n_reports: float) -> float:
        d = n_reports / span
        old = adw / violent
        new = max(0.0, adw - d) / max(1.0, violent - d)
        return (old - new) * 100.0

    theft_exit_pm = DOCUMENTED_PATH_FLOORS["theft_exit"] / span
    mde = {c["key"]: (1.645 * c["its"]["se_nw"] * 100.0)
           for c in directional if c.get("its")}
    doc_robbery_pp = robbery_pp(DOCUMENTED_PATH_FLOORS["robbery"])
    doc_adw_pp = adw_pp(DOCUMENTED_PATH_FLOORS["adw"])
    below = (doc_robbery_pp < mde.get("robbery_theft", 0.0)
             and doc_adw_pp < mde.get("adw_violent", 0.0))
    return {
        "documented_reports": DOCUMENTED_EDITED_REPORTS,
        "official_counts": DOCUMENTED_OFFICIAL_COUNTS,
        "path_floors": dict(DOCUMENTED_PATH_FLOORS),
        "era_months": span,
        "per_month": DOCUMENTED_EDITED_REPORTS / span,
        "monthly_volumes": {"robbery": rob, "theft_other": theft, "adw": adw,
                            "violent": violent, "total": total},
        # Documented per-path scenarios (the report's own counts).
        "doc_robbery_pp": doc_robbery_pp,
        "doc_adw_pp": doc_adw_pp,
        # Theft exits leave the feed entirely: an invisible fraction, not a
        # visible share shift. Reported as reports/month and % of the feed.
        "theft_exit_per_month": theft_exit_pm,
        "theft_exit_feed_pct": theft_exit_pm / total * 100.0 if total else 0.0,
        # Stress ceiling: every one of the 922 edits on a single path.
        "ceiling_robbery_pp": robbery_pp(DOCUMENTED_EDITED_REPORTS),
        "ceiling_adw_pp": adw_pp(DOCUMENTED_EDITED_REPORTS),
        "mde_pp": mde,
        "below_detectability": below,
    }


# Binseg has monthly grain and 12-month minimum segments, so a break caused
# by a band boundary can be placed a month or two BEFORE it. Breaks that
# close to a boundary get the boundary's attribution treatment, not a
# misleading nearest-dashed-line label.
BOUNDARY_ADJACENT_MONTHS = 2


def classify_break(b: date) -> str:
    """Which attribution bucket a detected break month falls in: 'covid'
    (inside the band), 'covid_boundary' (just before it), 'surge_era' (at or
    past the surge), 'surge_boundary' (just before it), or 'other'."""
    covid_lo, covid_hi = COVID_BAND
    if covid_lo <= b < covid_hi:
        return "covid"
    if add_months(covid_lo, -BOUNDARY_ADJACENT_MONTHS) <= b < covid_lo:
        return "covid_boundary"
    if b >= SURGE_START:
        return "surge_era"
    if add_months(SURGE_START, -BOUNDARY_ADJACENT_MONTHS) <= b:
        return "surge_boundary"
    return "other"


# ── Pure math: Benjamini-Hochberg ─────────────────────────────────────────────
def benjamini_hochberg(pvals: Sequence[float], q: float = BH_Q) -> dict:
    """Step-up FDR control. Returns per-test discovery flags (original order)
    and BH-adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adjusted[i] = running
    cutoff_rank = 0
    for rank in range(1, m + 1):
        if pvals[order[rank - 1]] <= q * rank / m:
            cutoff_rank = rank
    discoveries = [False] * m
    for rank in range(cutoff_rank):
        discoveries[order[rank]] = True
    return {"discoveries": discoveries, "adjusted": adjusted,
            "family_size": m, "q": q}


# ── Assembly (pure: matrix -> full result) ────────────────────────────────────
def _series_entry(key: str, label: str, kind: str, y: list[Optional[float]],
                  months: list[date], counts: Optional[list[int]] = None) -> dict:
    complete = [v if v is not None else 0.0 for v in y]
    deseason = deseasonalize(complete, months)
    breaks = binary_segmentation(deseason)
    its = its_level_shift(y, months)
    return {
        "key": key, "label": label, "kind": kind,
        "shares": y, "counts": counts,
        "breaks": [months[i] for i in breaks],
        "its": its,
    }


def classification_mix_from_matrix(matrix: dict) -> dict:
    """Pure assembly: the share matrix -> series, breaks, ITS + BH family."""
    months = matrix["months"]
    labels = {
        "THEFT/OTHER": "Theft (other)", "THEFT F/AUTO": "Theft from auto",
        "MOTOR VEHICLE THEFT": "Motor vehicle theft", "ROBBERY": "Robbery",
        "BURGLARY": "Burglary",
        "ASSAULT W/DANGEROUS WEAPON": "Assault w/ dangerous weapon",
        "SEX ABUSE": "Sex abuse", "HOMICIDE": "Homicide", "ARSON": "Arson",
    }
    series: list[dict] = []
    for off in VICTIM_OFFENSES:
        series.append(_series_entry(
            off, labels[off], "category",
            matrix["shares"][off], months, matrix["counts"][off]))
    for p in PAIRS:
        series.append(_series_entry(
            p["key"], p["label"], "pair",
            pair_share(matrix["counts"], p["a"], p["b"]), months))

    tested = [s for s in series if s["its"] is not None]
    bh = benjamini_hochberg([s["its"]["p_nw"] for s in tested])
    for s, disc, adj in zip(tested, bh["discoveries"], bh["adjusted"]):
        s["its"]["discovery"] = disc
        s["its"]["p_adjusted"] = adj

    # Specificity: the identical family test at placebo boundaries.
    placebos = []
    for pb in PLACEBO_BREAKS:
        ps = []
        for s in series:
            r = its_level_shift(s["shares"], months, break_month=pb)
            if r is not None:
                ps.append(r["p_nw"])
        if ps:
            pbh = benjamini_hochberg(ps)
            placebos.append({"month": pb,
                             "discoveries": sum(pbh["discoveries"]),
                             "family_size": pbh["family_size"]})

    for s in series:
        buckets: dict[str, list[date]] = {
            "covid": [], "covid_boundary": [], "surge_era": [],
            "surge_boundary": [], "other": []}
        for b in s["breaks"]:
            buckets[classify_break(b)].append(b)
        s["covid_breaks"] = buckets["covid"]
        s["covid_adjacent_breaks"] = buckets["covid_boundary"]
        s["surge_breaks"] = buckets["surge_era"]
        s["surge_adjacent_breaks"] = buckets["surge_boundary"]

    # ── Question 2: the documented manipulation era ───────────────────────────
    by_key = {s["key"]: s for s in series}
    era_tested = []
    for s in series:
        r = its_level_shift(s["shares"], months, break_month=ERA_BREAK,
                            window_months=ERA_WINDOW_MONTHS)
        s["era_its"] = r
        if r is not None:
            era_tested.append(s)
    era_family = 0
    if era_tested:
        era_bh = benjamini_hochberg([s["era_its"]["p_nw"] for s in era_tested])
        era_family = era_bh["family_size"]
        for s, disc, adj in zip(era_tested, era_bh["discoveries"],
                                era_bh["adjusted"]):
            s["era_its"]["discovery"] = disc
            s["era_its"]["p_adjusted"] = adj

    era_placebos = []
    for pb in ERA_PLACEBO_BREAKS:
        ps = []
        for s in series:
            r = its_level_shift(s["shares"], months, break_month=pb,
                                window_months=ERA_WINDOW_MONTHS)
            if r is not None:
                ps.append(r["p_nw"])
        if ps:
            pbh = benjamini_hochberg(ps)
            era_placebos.append({"month": pb,
                                 "discoveries": sum(pbh["discoveries"]),
                                 "family_size": pbh["family_size"]})

    # Directional family: one-sided, signs fixed a-priori from the IA report's
    # documented downgrade paths. Its own BH family, stated on the page.
    directional = []
    for c in DIRECTIONAL_CONTRASTS:
        s = by_key[c["key"]]
        its = s.get("era_its")
        entry: dict[str, Any] = {"key": c["key"], "label": s["label"],
                                 "sign": c["sign"], "path": c["path"],
                                 "its": its}
        if its is not None:
            entry["p_one"] = one_sided_p(its["shift"], its["se_nw"], c["sign"])
        directional.append(entry)
    d_tested = [e for e in directional if e.get("p_one") is not None]
    if d_tested:
        dbh = benjamini_hochberg([e["p_one"] for e in d_tested])
        for e, disc, adj in zip(d_tested, dbh["discoveries"], dbh["adjusted"]):
            e["discovery"] = disc
            e["p_adjusted"] = adj

    # Reversion family: same contrasts, post-exposure step, its own BH family.
    reversion = []
    for c in DIRECTIONAL_CONTRASTS:
        s = by_key[c["key"]]
        r = era_reversion(s["shares"], months, c["sign"])
        entry = {"key": c["key"], "label": s["label"], "rev": r}
        if r is not None:
            entry["p_one"] = r["p_one"]
        reversion.append(entry)
    r_tested = [e for e in reversion if e.get("p_one") is not None]
    if r_tested:
        rbh = benjamini_hochberg([e["p_one"] for e in r_tested])
        for e, disc, adj in zip(r_tested, rbh["discoveries"], rbh["adjusted"]):
            e["discovery"] = disc
            e["p_adjusted"] = adj
    rev_by_key = {e["key"]: e for e in reversion}
    for e in directional:
        e["reversion_discovery"] = bool(
            rev_by_key.get(e["key"], {}).get("discovery"))
        # The manipulation signature: decline under pressure, reversion after
        # exposure. Both legs corrected within their families.
        e["signature"] = bool(e.get("discovery")) and e["reversion_discovery"]

    era = {
        "break_month": ERA_BREAK,
        "window_months": ERA_WINDOW_MONTHS,
        "family_size": era_family,
        "placebos": era_placebos,
        "directional": directional,
        "reversion": reversion,
        "magnitude": magnitude_bridge(matrix, months, directional),
        "surge_start": SURGE_START,
        "exposure_month": EXPOSURE_MONTH,
        "documented_reports": DOCUMENTED_EDITED_REPORTS,
    }

    training_idx = month_index(months, EVENTS[-1]["month"])
    return {
        "meta": {
            "definition": CLASSIFICATION_DEFINITION,
            "window": {"start": months[0], "end_exclusive": add_months(months[-1], 1),
                       "months": len(months)},
            "events": EVENTS,
            "covid_band": COVID_BAND,
            "covid_label": COVID_LABEL,
            "surge_band": (SURGE_START, add_months(months[-1], 1)),
            "surge_label": SURGE_LABEL,
            "break_month": BREAK_MONTH,
            "its_window_months": ITS_WINDOW_MONTHS,
            "placebos": placebos,
            "family_size": bh["family_size"],
            "bh_q": bh["q"],
            "nw_lags": next((s["its"]["nw_lags"] for s in tested), None),
            "min_segment_months": MIN_SEGMENT_MONTHS,
            "penalty_mult": BINSEG_PENALTY_MULT,
            "post_training_months": (len(months) - training_idx
                                     if training_idx is not None else 0),
        },
        "months": months,
        "totals": matrix["totals"],
        "series": series,
        "era": era,
    }


# ── DB access ─────────────────────────────────────────────────────────────────
_MONTHLY_OFFENSE_SQL = """
SELECT date_trunc('month', report_dat AT TIME ZONE 'UTC')::date AS month,
       offense, COUNT(*)::int AS n
FROM crimes
WHERE city='dc' AND offense IS NOT NULL
  AND CAST(report_dat AT TIME ZONE 'UTC' AS date) >= $1
  AND CAST(report_dat AT TIME ZONE 'UTC' AS date) < $2
GROUP BY 1, 2
ORDER BY 1
"""


async def classification_mix(conn, today: Optional[date] = None) -> dict:
    """Monthly classification-mix analysis over the full archive, through the
    last complete month."""
    end_exclusive = month_start(today or date.today())
    rows = await conn.fetch(_MONTHLY_OFFENSE_SQL, ARCHIVE_START, end_exclusive)
    if not rows:
        raise RuntimeError("no incident archive - run the crime ingest first.")
    matrix = share_matrix(
        [{"month": r["month"], "offense": r["offense"], "n": r["n"]} for r in rows],
        ARCHIVE_START, end_exclusive)
    if sum(matrix["totals"]) == 0:
        raise RuntimeError("incident archive is empty in the analysis window.")
    return classification_mix_from_matrix(matrix)
