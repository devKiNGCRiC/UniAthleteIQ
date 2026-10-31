"""Statistical inference for model evaluation.

Why a *cluster* (athlete-level) bootstrap?
    Rows from the same athlete are strongly correlated: consecutive days share
    6 of their 7 days of history. Resampling individual rows would treat
    ~35,000 athlete-days as independent evidence and produce confidence
    intervals that are far too narrow. Instead we resample whole athletes with
    replacement, so the effective sample size is the number of athletes (tens),
    which is the honest unit of generalisation.

Two sources of uncertainty are reported separately:
    * split variability    - how much a metric moves between different random
                             CV fold assignments (repeated CV, one value per seed)
    * sampling variability - how much it would move with a different set of
                             athletes (athlete bootstrap of the seed-averaged
                             out-of-fold predictions)

Speed trick:
    Drawing athletes with replacement is equivalent to giving every row an
    integer *weight* (how many times its athlete was drawn). ROC-AUC and
    average precision can be computed from weighted cumulative sums over rows
    sorted by score, so we sort once and each bootstrap round is O(n) with no
    re-sorting. ``_weighted_metrics`` reproduces sklearn's ``roc_auc_score``
    and ``average_precision_score`` exactly (verified in tests/test_stats.py).
"""
import numpy as np
import pandas as pd

METRIC_NAMES = ("roc_auc", "pr_auc")


class _SortedScores:
    """Scores pre-sorted (descending) with tie blocks marked, reusable across bootstrap rounds."""

    def __init__(self, y, p, groups=None):
        order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
        self.order = order
        # Athlete code of every row in sorted order, so a bootstrap draw (count per
        # athlete) turns into per-row weights with a single fancy-index.
        self.codes = None if groups is None else _athlete_codes(groups)[0][order]
        self.y = np.asarray(y, dtype=float)[order]
        p_sorted = np.asarray(p, dtype=float)[order]
        # Index of the LAST row of each block of tied scores: metrics may only
        # change at a threshold where the score value actually changes.
        self.block_ends = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]

    def metrics(self, w=None):
        """Weighted ROC-AUC and average precision. ``w`` is a per-row weight in ORIGINAL order."""
        w = np.ones_like(self.y) if w is None else np.asarray(w, dtype=float)[self.order]
        return _weighted_metrics(self.y, w, self.block_ends)

    def metrics_from_draws(self, draws):
        """Metrics when athlete ``i`` was drawn ``draws[i]`` times (bootstrap round)."""
        return _weighted_metrics(self.y, draws[self.codes].astype(float), self.block_ends)


def _weighted_metrics(y_sorted, w_sorted, block_ends):
    # Cumulative weighted true/false positives when flagging everything at or above each threshold
    tp = np.cumsum(w_sorted * y_sorted)[block_ends]
    fp = np.cumsum(w_sorted * (1 - y_sorted))[block_ends]
    pos, neg = tp[-1], fp[-1]
    if pos == 0 or neg == 0:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    # ROC-AUC: trapezoidal area under (FP, TP) steps; trapezoids give tied scores half credit
    tp0, fp0 = np.r_[0.0, tp], np.r_[0.0, fp]
    auc = np.sum(np.diff(fp0) * (tp0[1:] + tp0[:-1]) / 2) / (pos * neg)
    # Average precision (sklearn's definition): sum over thresholds of recall gain x precision
    # Thresholds above every drawn row have tp + fp == 0; they add no recall, so precision 0 is safe there.
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall_gain = np.diff(np.r_[0.0, tp]) / pos
    ap = np.sum(recall_gain * precision)
    return {"roc_auc": auc, "pr_auc": ap}


def point_metrics(y, p):
    """Unweighted ROC-AUC and PR-AUC (identical to sklearn's)."""
    return _SortedScores(y, p).metrics()


def _athlete_codes(groups):
    codes, uniques = pd.factorize(np.asarray(groups))
    return codes, len(uniques)


def athlete_bootstrap_draws(groups, n_boot=1000, seed=0):
    """Yield, per bootstrap round, how many times each athlete was drawn (with replacement)."""
    _, n_athletes = _athlete_codes(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n_athletes, size=n_athletes), minlength=n_athletes)


def athlete_bootstrap_weights(groups, n_boot=1000, seed=0):
    """Per-row weights for each round: an athlete drawn twice weighs 2 on all their rows.

    This is exactly equivalent to duplicating that athlete's rows (standard
    cluster bootstrap); see tests/test_stats.py.
    """
    codes, _ = _athlete_codes(groups)
    for draws in athlete_bootstrap_draws(groups, n_boot, seed):
        yield draws[codes]


def _percentile_ci(values, alpha):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]  # a resample with no injuries has undefined metrics
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, len(values)


def bootstrap_ci(y, p, groups, n_boot=1000, alpha=0.05, seed=0):
    """Point estimates and athlete-bootstrap percentile CIs for ROC-AUC and PR-AUC.

    Returns ``{"roc_auc": {"estimate", "ci_low", "ci_high", "n_boot"}, "pr_auc": {...}}``.
    """
    s = _SortedScores(y, p, groups)
    boots = [s.metrics_from_draws(d) for d in athlete_bootstrap_draws(groups, n_boot, seed)]
    point = s.metrics()
    out = {}
    for m in METRIC_NAMES:
        lo, hi, n = _percentile_ci([b[m] for b in boots], alpha)
        out[m] = {"estimate": point[m], "ci_low": lo, "ci_high": hi, "n_boot": n}
    return out


def paired_bootstrap(y, p_a, p_b, groups, n_boot=1000, alpha=0.05, seed=0):
    """Compare two models evaluated on the *same* rows (A minus B), for both metrics.

    Both models are scored on the identical resampled set of athletes in every
    bootstrap iteration, so athlete-to-athlete difficulty cancels out and the
    test is far more sensitive than comparing two separate CIs.

    For each metric returns the observed difference, its CI, and a two-sided
    bootstrap p-value (twice the share of resamples on the "wrong" side of zero).
    """
    a, b = _SortedScores(y, p_a, groups), _SortedScores(y, p_b, groups)
    diffs = {m: [] for m in METRIC_NAMES}
    for d in athlete_bootstrap_draws(groups, n_boot, seed):
        ma, mb = a.metrics_from_draws(d), b.metrics_from_draws(d)
        for m in METRIC_NAMES:
            diffs[m].append(ma[m] - mb[m])
    pa, pb = a.metrics(), b.metrics()
    out = {}
    for m in METRIC_NAMES:
        dm = np.asarray(diffs[m])
        dm = dm[~np.isnan(dm)]
        lo, hi, n = _percentile_ci(dm, alpha)
        p_value = min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))
        out[m] = {"difference": pa[m] - pb[m], "ci_low": lo, "ci_high": hi, "p_value": p_value, "n_boot": n}
    return out


def holm_adjust(p_values):
    """Holm-Bonferroni correction for multiple comparisons (controls family-wise error).

    Less conservative than plain Bonferroni but still valid under any dependence
    between the tests, which matters here because all comparisons reuse the
    same athletes.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def seed_spread(y, oof_by_seed, metric="roc_auc"):
    """Metric per CV repetition -> mean and SD (split variability)."""
    vals = np.array([point_metrics(y, p)[metric] for p in oof_by_seed])
    return {"mean": vals.mean(), "sd": vals.std(ddof=1) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
