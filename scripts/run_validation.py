"""Heavy experiments for the statistical-validation study (notebook 03).

Runs every model/feature-set configuration with repeated athlete-grouped CV,
plus baselines and sensitivity analyses, and caches all out-of-fold
predictions to ``outputs/cache/validation.joblib``. The notebook then does
the (fast) statistics and plotting from that cache, so it can be re-run and
edited without repeating ~30 minutes of model fitting.

Usage:
    python scripts/run_validation.py            # full run (10 seeds)
    python scripts/run_validation.py --seeds 2  # quick smoke run
"""
import argparse
import time
import warnings

import joblib
import numpy as np

from uniathleteiq import baselines, data, evaluation as ev, models, paths

warnings.filterwarnings("ignore")  # sklearn/xgboost convergence chatter would bury the progress log

CACHE = paths.OUTPUT_DIR / "cache" / "validation.joblib"
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main(n_seeds):
    t0 = time.time()
    seeds = list(range(n_seeds))
    out = {"seeds": seeds, "runners": {}, "football": {}}

    # ── Runners ─────────────────────────────────────────────────────────────
    # The same athlete hold-out as notebook 02 (seed 42): the 15 test athletes are
    # never touched by any of the repeated-CV experiments below.
    ref = data.load_runners("engineered+personal")
    train_idx, test_idx = ev.holdout_split(ref)
    r = out["runners"]
    r["y"] = ref.y.iloc[train_idx].to_numpy()
    r["groups"] = ref.groups.iloc[train_idx].to_numpy()
    r["oof"] = {}

    # 1) Main grid: feature set x model, repeated CV
    for fs in data.RUNNER_FEATURE_SETS:
        ds = data.load_runners(fs)
        X, y, g = ds.X.iloc[train_idx], ds.y.iloc[train_idx], ds.groups.iloc[train_idx]
        zoo = models.make_models(y, include_smote=False)
        for name in MODEL_NAMES:
            r["oof"][(fs, name)] = ev.repeated_grouped_oof(zoo[name], X, y, g, seeds)
            log(f"runners  {fs:20s} {name}", t0)

    # 2) Untrained baselines on the same training athletes (no CV needed: nothing is fitted)
    r["baselines"] = baselines.runner_baselines(ref.X.iloc[train_idx]).to_dict("series")
    r["baselines"] = {k: v.to_numpy() for k, v in r["baselines"].items()}

    # 3) Sensitivity: how much history is required before personal z-scores kick in
    r["sensitivity_min_history"] = {}
    for k in (14, 28, 56):
        ds = data.load_runners("engineered+personal", personal_min_history=k)
        X, y, g = ds.X.iloc[train_idx], ds.y.iloc[train_idx], ds.groups.iloc[train_idx]
        lr = models.make_models(y, include_smote=False)["Logistic Regression"]
        r["sensitivity_min_history"][k] = ev.repeated_grouped_oof(lr, X, y, g, seeds)
        log(f"runners  sensitivity min_history={k}", t0)

    # 4) Final held-out test of the chosen model, plus baselines on the same athletes
    lr = models.make_models(ref.y.iloc[train_idx], include_smote=False)["Logistic Regression"]
    lr.fit(ref.X.iloc[train_idx], ref.y.iloc[train_idx])
    r["holdout"] = {
        "y": ref.y.iloc[test_idx].to_numpy(),
        "groups": ref.groups.iloc[test_idx].to_numpy(),
        "model": lr.predict_proba(ref.X.iloc[test_idx])[:, 1],
        "baselines": {k: v.to_numpy() for k, v in baselines.runner_baselines(ref.X.iloc[test_idx]).items()},
    }
    log("runners  hold-out", t0)

    # ── Football: sensitivity of conclusions to the "severe injury" definition ──
    for days in (14, 28, 56):
        fb = data.load_football(severe_days=days)
        ftr, _ = ev.holdout_split(fb)
        X, y, g = fb.X.iloc[ftr], fb.y.iloc[ftr], fb.groups.iloc[ftr]
        zoo = models.make_models(y, include_smote=False)
        out["football"][days] = {
            "y": y.to_numpy(), "groups": g.to_numpy(),
            "oof": {name: ev.repeated_grouped_oof(zoo[name], X, y, g, seeds) for name in MODEL_NAMES},
            # Simplest football baseline: days injured last season (0 when no history)
            "baseline_prev_season_days": X["season_days_injured_prev_season"].fillna(0).to_numpy(),
        }
        log(f"football severe > {days} days", t0)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, CACHE, compress=3)
    log(f"saved {CACHE}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of repeated-CV seeds (default 10)")
    main(ap.parse_args().seeds)
