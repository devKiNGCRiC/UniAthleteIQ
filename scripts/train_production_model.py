"""Train the model the web app serves.

Research notebooks hold out 15 athletes to measure honest performance. A
deployed model should learn from *all* available athletes, so this script:

1. Runs athlete-grouped CV on all 74 athletes to get out-of-fold predictions
   (used to pick the alert threshold and fit the probability calibration
   without ever scoring an athlete the model was trained on).
2. Fits the final Logistic Regression on all athletes.
3. Saves a versioned bundle the API loads at startup.

Usage:
    python scripts/train_production_model.py
"""
import datetime as dt
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from uniathleteiq import __version__, data, evaluation as ev, models, paths

warnings.filterwarnings("ignore")

OUT = paths.MODELS_DIR / "production" / "runner_model.joblib"
TARGET_RECALL = 0.5  # alert threshold: flag enough days to catch half of injuries


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    ds = data.load_runners("engineered+personal")
    model = models.make_models(ds.y, include_smote=False)["Logistic Regression"]

    # Out-of-fold predictions: each athlete scored by a model that never saw them
    oof, fold_auc = ev.grouped_oof(model, ds.X, ds.y, ds.groups)
    cv = ev.score(ds.y, oof)
    threshold = ev.threshold_for_recall(ds.y, oof, TARGET_RECALL)
    # Platt scaling turns the class-weighted score into a calibrated injury probability
    platt = LogisticRegression().fit(logit(oof), ds.y)

    model.fit(ds.X, ds.y)
    bundle = {
        "model": model,
        "features": list(ds.X.columns),
        "threshold": float(threshold),
        "platt": platt,
        "base_rate": float(ds.y.mean()),
        "metadata": {
            "version": f"runner-lr-{__version__}",
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_athletes": int(ds.groups.nunique()),
            "n_athlete_days": int(len(ds.y)),
            "n_injuries": int(ds.y.sum()),
            "cv_roc_auc": float(cv["roc_auc"]),
            "cv_pr_auc": float(cv["pr_auc"]),
            "cv_roc_auc_fold_sd": float(fold_auc.std()),
            "target_recall": TARGET_RECALL,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"Saved {OUT}")
    for k, v in bundle["metadata"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
