"""Dataset loaders and feature engineering.

Each loader returns a ``Dataset`` with features, a binary target and a
group id (athlete / player) so that evaluation can keep every athlete
entirely in train or entirely in test.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    meta: pd.DataFrame | None = None


# ── Runners (Lövdal et al., competitive runners, 74 athletes) ────────────────
#
# Day file: 7 days of history per row. Suffix "" is the oldest day, ".6" is the
# most recent day. Week file: 3 weeks of aggregates, week 0 ("" suffix) is the
# most recent. Perceived scores are -0.01 on rest days (no session).

DAY_VARS = ["nr. sessions", "total km", "km Z3-4", "km Z5-T1-T2", "km sprinting",
            "strength training", "hours alternative", "perceived exertion",
            "perceived trainingSuccess", "perceived recovery"]
PERCEIVED = ["perceived exertion", "perceived trainingSuccess", "perceived recovery"]
WEEK_PERCEIVED = ["avg exertion", "min exertion", "max exertion",
                  "avg training success", "min training success", "max training success",
                  "avg recovery", "min recovery", "max recovery"]


def _day_col(var, day):
    return var if day == 0 else f"{var}.{day}"


def _week_col(var, week):
    return var if week == 0 else f"{var}.{week}"


def _daily(df, var):
    """(n_rows, 7) array of a daily variable, oldest day first."""
    return df[[_day_col(var, d) for d in range(7)]].to_numpy(dtype=float)


def runner_engineered_features(df):
    """Workload-science features: ACWR, monotony, strain, recent load, wellness."""
    km = _daily(df, "total km")
    sessions = _daily(df, "nr. sessions")
    hard_km = _daily(df, "km Z5-T1-T2") + _daily(df, "km sprinting")
    exertion, recovery, success = (np.where(_daily(df, v) < 0, np.nan, _daily(df, v))
                                   for v in PERCEIVED)

    weekly_km = np.column_stack([df[_week_col("total kms", w)] for w in range(3)])
    acute = weekly_km[:, 0]
    chronic = weekly_km.mean(axis=1)
    km_mean, km_std = km.mean(axis=1), km.std(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        feats = pd.DataFrame({
            "km_7d": km.sum(axis=1),
            "km_last_3d": km[:, 4:].sum(axis=1),
            "km_yesterday": km[:, 6],
            "hard_km_7d": hard_km.sum(axis=1),
            "rest_days_7d": (sessions == 0).sum(axis=1),
            "sessions_7d": sessions.sum(axis=1),
            "monotony_7d": np.where(km_std > 0, km_mean / np.where(km_std > 0, km_std, 1), np.nan),
            "acwr_3wk": np.where(chronic > 0, acute / np.where(chronic > 0, chronic, 1), np.nan),
            "km_change_vs_prev_week": acute - weekly_km[:, 1],
            # Week-over-week load ratios on a log scale with +1 km smoothing. The dataset's own
            # "rel total kms" columns divide by (km + 1e-6), which explodes to ~2e8 after a
            # zero-km week and wrecks linear models; log((a+1)/(b+1)) is bounded and symmetric.
            "km_ratio_week0_vs_1": np.log((weekly_km[:, 0] + 1) / (weekly_km[:, 1] + 1)),
            "km_ratio_week0_vs_2": np.log((weekly_km[:, 0] + 1) / (weekly_km[:, 2] + 1)),
            "km_ratio_week1_vs_2": np.log((weekly_km[:, 1] + 1) / (weekly_km[:, 2] + 1)),
            "exertion_mean_7d": np.nanmean(exertion, axis=1),
            "exertion_max_7d": np.nanmax(exertion, axis=1),
            "exertion_trend": np.nanmean(exertion[:, 4:], axis=1) - np.nanmean(exertion[:, :4], axis=1),
            "recovery_mean_7d": np.nanmean(recovery, axis=1),
            "recovery_min_7d": np.nanmin(recovery, axis=1),
            "recovery_trend": np.nanmean(recovery[:, 4:], axis=1) - np.nanmean(recovery[:, :4], axis=1),
            "training_success_mean_7d": np.nanmean(success, axis=1),
            "exertion_minus_recovery_7d": np.nanmean(exertion, axis=1) - np.nanmean(recovery, axis=1),
        }, index=df.index)
    feats["strain_7d"] = feats["km_7d"] * feats["monotony_7d"]
    return feats


PERSONAL_BASE = ["km_7d", "km_last_3d", "km_yesterday", "hard_km_7d", "sessions_7d", "rest_days_7d",
                 "monotony_7d", "strain_7d", "acwr_3wk", "exertion_mean_7d", "recovery_mean_7d",
                 "training_success_mean_7d", "exertion_minus_recovery_7d"]
PERSONAL_MIN_HISTORY = 28  # days of an athlete's own history before a personal baseline exists
PERSONAL_STD_FLOOR = 0.25  # x |past mean|, so near-constant early histories don't explode
PERSONAL_CLIP = 4.0


def personal_baseline_features(feats, athlete, date, cols=PERSONAL_BASE, min_history=PERSONAL_MIN_HISTORY):
    """z-score of each feature against the athlete's OWN past (strictly earlier days only).

    Before an athlete has ``min_history`` days, or when the feature is undefined,
    the score is 0 ("at their usual level"). Deliberately no missing-indicator:
    in this dataset the injury rate depends on how long an athlete has been
    recorded, and an indicator would let the model learn that collection artifact.
    """
    f = feats[cols].copy()
    f["_a"], f["_d"] = athlete.values, date.values
    f = f.sort_values(["_a", "_d"])
    out = {}
    for c in cols:
        g = f.groupby("_a")[c]
        past_mean = g.transform(lambda s: s.expanding(min_periods=min_history).mean().shift(1))
        past_std = g.transform(lambda s: s.expanding(min_periods=min_history).std().shift(1))
        # Floor from the athlete's own past only (a dataset-wide SD would include future days)
        past_std = np.maximum(past_std, np.maximum(PERSONAL_STD_FLOOR * past_mean.abs(), 1e-3))
        z = ((f[c] - past_mean) / past_std).clip(-PERSONAL_CLIP, PERSONAL_CLIP)
        out[f"{c}_vs_personal"] = z.fillna(0.0)
    return pd.DataFrame(out).loc[feats.index]


RUNNER_FEATURE_SETS = ("raw", "raw+engineered", "engineered", "engineered+personal")


def load_runners(feature_set="engineered+personal", personal_min_history=PERSONAL_MIN_HISTORY):
    """Runner athlete-days.

    feature_set:
      raw                  - the 70 daily + 69 weekly columns as provided
      raw+engineered       - raw plus workload-science features
      engineered           - workload-science features + week-over-week ratios
      engineered+personal  - engineered plus deviation from the athlete's own baseline (final model)
    personal_min_history:
      days of an athlete's own history required before personal z-scores are
      computed (exposed for sensitivity analysis; default 28).
    """
    import warnings

    if feature_set not in RUNNER_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {RUNNER_FEATURE_SETS}")
    day = pd.read_csv(paths.RUNNERS_DAY_CSV)
    week = pd.read_csv(paths.RUNNERS_WEEK_CSV)
    df = day.merge(week.drop(columns="injury"), on=["Athlete ID", "Date"], how="inner",
                   suffixes=("", "_wk"))
    df = df.sort_values(["Athlete ID", "Date"]).reset_index(drop=True)

    # Readable, unique names: "<var> (Nd ago)" for daily, "<var> (week -N)" for weekly
    names = {}
    for d in range(7):
        for v in DAY_VARS:
            names[_day_col(v, d)] = f"{v} ({7 - d}d ago)"
    for c in week.columns:
        if c in ("Athlete ID", "injury", "Date"):
            continue
        merged = f"{c}_wk" if c in day.columns else c
        base, _, w = c.rpartition(".")
        label = f"{base} (week -{w})" if w.isdigit() and base else f"{c} (week -0)"
        names[merged] = c if c.startswith("rel total kms") else label
    X = df[list(names)].astype(float).rename(columns=names)
    # Rest-day sentinel (-0.01) -> missing
    perceived_cols = [names[_day_col(v, d)] for v in PERCEIVED for d in range(7)]
    perceived_cols += [n for n in X.columns if any(n.startswith(v + " (week") for v in WEEK_PERCEIVED)]
    X[perceived_cols] = X[perceived_cols].mask(X[perceived_cols] < 0)
    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_set != "raw":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-rest weeks -> NaN means
            eng = runner_engineered_features(df)
        # "raw" columns are kept exactly as provided (including the unstable "rel total kms"
        # ratios) so the ablation measures the dataset as published; the engineered sets
        # use the bounded log-ratios from runner_engineered_features instead.
        X = pd.concat([X, eng], axis=1) if feature_set == "raw+engineered" else eng
        if feature_set == "engineered+personal":
            X = pd.concat([X, personal_baseline_features(
                eng, df["Athlete ID"], df["Date"], min_history=personal_min_history)], axis=1)

    return Dataset("Runners", X, df["injury"].astype(int), df["Athlete ID"],
                   meta=df[["Athlete ID", "Date"]])


# ── Football (EPL players, one row per player-season) ────────────────────────
#
# Only information known at the START of the season is used. Columns describing
# the season being predicted or the whole career (which includes future seasons)
# are excluded because they leak the outcome.

FOOTBALL_LEAKY = ["season_days_injured", "total_days_injured", "season_minutes_played",
                  "season_games_played", "season_matches_in_squad",
                  "total_minutes_played", "total_games_played"]
FOOTBALL_FEATURES = ["age", "height_cm", "weight_kg", "bmi", "work_rate_numeric",
                     "position_numeric", "pace", "physic", "fifa_rating",
                     "cumulative_minutes_played", "cumulative_games_played",
                     "minutes_per_game_prev_seasons", "avg_days_injured_prev_seasons",
                     "avg_games_per_season_prev_seasons", "significant_injury_prev_season",
                     "cumulative_days_injured", "season_days_injured_prev_season"]
# Time-loss injury severity convention (Fuller et al., 2006): severe = more than 28 days.
SEVERE_DAYS = 28


def load_football(severe_days=SEVERE_DAYS):
    df = pd.read_csv(paths.FOOTBALL_CSV)
    X = df[FOOTBALL_FEATURES].astype(float).copy()
    X["has_injury_history"] = df["cumulative_days_injured"].notna().astype(float)
    y = (df["season_days_injured"] > severe_days).astype(int)
    return Dataset("Football", X, y, df["p_id2"], meta=df[["p_id2", "start_year"]])


# ── Multi-sport wearable dataset (1,000 sessions, one per athlete) ───────────

def load_multisport():
    df = pd.read_csv(paths.MULTISPORT_CSV)
    num = ["Heart_Rate_BPM", "Respiratory_Rate_BPM", "Skin_Temperature_C",
           "Blood_Oxygen_Level_Percent", "Impact_Force_Newtons", "Cumulative_Fatigue_Index",
           "Duration_Minutes", "Injury_Risk_Score"]
    X = pd.concat([df[num].astype(float),
                   pd.get_dummies(df[["Sport_Type", "Activity_Type"]], dtype=float)], axis=1)
    return Dataset("Multi-sport (wearable)", X, df["Injury_Occurred"].astype(int),
                   df["Athlete_ID"], meta=df[["Athlete_ID", "Sport_Type"]])


# ── Shared, dataset-normalised features for the cross-sport experiment ───────

def harmonised_load_effort():
    """Two concepts every session-level dataset has: training load and effort.

    Values are z-scored *within* each dataset so units (km vs minutes,
    RPE vs fatigue index) become comparable.
    """
    r = load_runners()
    m_raw = pd.read_csv(paths.MULTISPORT_CSV)

    def z(s):
        return (s - s.mean()) / s.std()

    runners = pd.DataFrame({"load_z": z(r.X["km_7d"]),
                            "effort_z": z(r.X["exertion_mean_7d"].fillna(r.X["exertion_mean_7d"].median())),
                            "injury": r.y, "group": r.groups, "dataset": "Runners"})
    multi = pd.DataFrame({"load_z": z(m_raw["Duration_Minutes"].astype(float)),
                          "effort_z": z(m_raw["Cumulative_Fatigue_Index"]),
                          "injury": m_raw["Injury_Occurred"], "group": m_raw["Athlete_ID"],
                          "dataset": "Multi-sport (wearable)"})
    return pd.concat([runners, multi], ignore_index=True)


# ── The original v1 harmonisation, kept to reproduce the leakage audit ──────

def load_v1_combined():
    multi = pd.read_csv(paths.MULTISPORT_CSV)
    day = pd.read_csv(paths.RUNNERS_DAY_CSV)
    week = pd.read_csv(paths.RUNNERS_WEEK_CSV)
    fb = pd.read_csv(paths.FOOTBALL_CSV)
    cols = ["heart_rate", "fatigue_index", "duration_minutes", "injury_risk_score",
            "injury_occurred", "sport"]
    parts = [
        pd.DataFrame({"heart_rate": multi["Heart_Rate_BPM"], "fatigue_index": multi["Cumulative_Fatigue_Index"],
                      "duration_minutes": multi["Duration_Minutes"], "injury_risk_score": multi["Injury_Risk_Score"],
                      "injury_occurred": multi["Injury_Occurred"], "sport": multi["Sport_Type"]}),
        pd.DataFrame({"heart_rate": np.nan, "fatigue_index": day["perceived exertion"],
                      "duration_minutes": day["total km"], "injury_risk_score": day["perceived recovery"],
                      "injury_occurred": day["injury"], "sport": "Athletics"}),
        pd.DataFrame({"heart_rate": np.nan, "fatigue_index": week["avg exertion"],
                      "duration_minutes": week["total kms"], "injury_risk_score": week["avg recovery"],
                      "injury_occurred": week["injury"], "sport": "Athletics"}),
        pd.DataFrame({"heart_rate": np.nan, "fatigue_index": np.nan,
                      "duration_minutes": fb["season_minutes_played"], "injury_risk_score": np.nan,
                      "injury_occurred": (fb["season_days_injured"] > 0).astype(int), "sport": "Football"}),
    ]
    df = pd.concat(parts, ignore_index=True)[cols]
    for c in ["heart_rate", "fatigue_index", "injury_risk_score", "duration_minutes"]:
        df[c] = df[c].fillna(df[c].median())
    df["sport_encoded"] = df["sport"].astype("category").cat.codes
    return df
