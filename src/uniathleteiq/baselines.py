"""Simple, untrained reference scores that any ML model must beat.

A reviewer's first question is "does the model do better than what coaches
already use?". These scores need no training, so they cannot overfit and are
evaluated on exactly the same rows as the ML models.
"""
import numpy as np
import pandas as pd

# Gabbett (2016) popularised an acute:chronic workload ratio above ~1.5 as a
# "danger zone". We use it both as a yes/no rule and as a continuous score.
ACWR_DANGER = 1.5


def runner_baselines(X):
    """Return a DataFrame of baseline risk scores (higher = riskier) for runner rows.

    Expects the engineered runner features from ``data.load_runners``.
    Missing values (e.g. ACWR with no chronic load) get the lowest score so
    they are never flagged as high risk.
    """
    acwr = X["acwr_3wk"]
    return pd.DataFrame({
        # Continuous ACWR: does "more spike = more risk" rank days well?
        "ACWR (continuous)": acwr.fillna(acwr.min()),
        # The binary rule coaches actually apply.
        f"ACWR > {ACWR_DANGER} rule": (acwr > ACWR_DANGER).astype(float),
        # Plain training volume: the simplest possible load signal.
        "7-day km": X["km_7d"].fillna(0.0),
        # Subjective effort: athletes' own sense of how hard the week was.
        "Mean perceived exertion (7d)": X["exertion_mean_7d"].fillna(X["exertion_mean_7d"].min()),
    }, index=X.index)


def rule_metrics(y, flag):
    """Precision / recall / flagged share of a yes-no rule."""
    y, flag = np.asarray(y), np.asarray(flag).astype(bool)
    tp = (flag & (y == 1)).sum()
    return {"precision": tp / max(flag.sum(), 1), "recall": tp / max(y.sum(), 1), "flagged_rate": flag.mean()}
