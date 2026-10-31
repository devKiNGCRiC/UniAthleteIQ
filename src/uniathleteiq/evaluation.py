"""Leakage-free evaluation: athlete-grouped splits, imbalance-aware metrics."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from .models import RANDOM_STATE


def holdout_split(ds, test_size=0.2, seed=RANDOM_STATE):
    """Hold out a set of *athletes* (never seen in training) for the final test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(ds.X, ds.y, ds.groups))
    return train_idx, test_idx


def grouped_oof(model, X, y, groups, n_splits=5, seed=RANDOM_STATE):
    """Out-of-fold probabilities with athletes kept within a single fold."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        if y.iloc[te].nunique() > 1:
            fold_auc.append(roc_auc_score(y.iloc[te], oof[te]))
    return oof, np.array(fold_auc)


def threshold_metrics(y, p, threshold):
    yp = (p >= threshold).astype(int)
    return {"precision": precision_score(y, yp, zero_division=0),
            "recall": recall_score(y, yp, zero_division=0),
            "f1": f1_score(y, yp, zero_division=0),
            "flagged_rate": yp.mean()}


def score(y, p):
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "pr_auc_baseline": float(np.mean(y)),
            "brier": brier_score_loss(y, p)}


def threshold_for_recall(y, p, target_recall):
    """Highest threshold that still catches at least ``target_recall`` of injuries."""
    for t in np.unique(p)[::-1]:
        if recall_score(y, (p >= t).astype(int)) >= target_recall:
            return float(t)
    return float(np.min(p))


def compare_models(models, X, y, groups, n_splits=5):
    rows, oofs = [], {}
    for name, model in models.items():
        oof, fold_auc = grouped_oof(model, X, y, groups, n_splits)
        oofs[name] = oof
        s = score(y, oof)
        rows.append({"model": name, **s,
                     "roc_auc_fold_mean": fold_auc.mean(), "roc_auc_fold_std": fold_auc.std()})
    return pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False), oofs


def leakage_audit(df, features, target="injury_occurred"):
    """Reproduce the v1 protocol and compare it with a correct protocol on the same data."""
    X, y = df[features], df[target]
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rows = []

    def add(name, y_true, p):
        yp = (p >= 0.5).astype(int)
        rows.append({"protocol": name, "accuracy": accuracy_score(y_true, yp),
                     "f1": f1_score(y_true, yp), "precision": precision_score(y_true, yp, zero_division=0),
                     "recall": recall_score(y_true, yp), "roc_auc": roc_auc_score(y_true, p),
                     "pr_auc": average_precision_score(y_true, p)})

    # v1: SMOTE on the full dataset, then split -> synthetic copies of test rows in train
    Xb, yb = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    a, b, c, d = train_test_split(Xb, yb, test_size=0.2, random_state=RANDOM_STATE, stratify=yb)
    add("v1: SMOTE before split (leaky)", d, clone(rf).fit(a, c).predict_proba(b)[:, 1])

    # Correct: deduplicate, split first, SMOTE on training rows only, test on real distribution
    dd = df.drop_duplicates(features + [target])
    Xd, yd = dd[features], dd[target]
    a, b, c, d = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE, stratify=yd)
    a2, c2 = SMOTE(random_state=RANDOM_STATE).fit_resample(a, c)
    add("Correct: dedupe, split, SMOTE on train only", d, clone(rf).fit(a2, c2).predict_proba(b)[:, 1])

    # Sanity check: the 'sport' column alone (dataset identity)
    m = clone(rf).fit(a2[["sport_encoded"]], c2)
    add("Sport column only (dataset identity)", d, m.predict_proba(b[["sport_encoded"]])[:, 1])
    return pd.DataFrame(rows).set_index("protocol")
