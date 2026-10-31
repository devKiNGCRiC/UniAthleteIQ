"""Turn an athlete's daily training logs into model features (used by the API).

Training/serving skew is the classic way a deployed model silently breaks: the
product computes a feature slightly differently from the training pipeline.
To rule that out, this module does NOT re-implement any feature logic. It
reshapes daily logs into the exact wide format of the research dataset
(``total km``, ``total km.1`` ... ``total kms.2``) and then calls the same
``runner_engineered_features`` and ``personal_baseline_features`` functions used
in training. ``tests/test_serving.py`` checks the result against the dataset.

Time convention (matches the research data):
    Risk for day D is computed from the 7 days D-7 .. D-1 (daily detail) and the
    21 days D-21 .. D-1 (three weekly totals). So once today's log is entered,
    the app can show the risk for *tomorrow*.

Days without a log are treated as rest days (0 km, no perceived scores). That is
the natural reading for a daily training diary, but it means athletes must log
every training day for the risk to be meaningful; the API reports how many of
the last 21 days were actually logged.
"""
import warnings

import numpy as np
import pandas as pd

from . import data

# API field name -> research-dataset column name. The API uses snake_case names;
# the order follows data.DAY_VARS.
LOG_FIELDS = {
    "sessions": "nr. sessions",
    "total_km": "total km",
    "km_z3_4": "km Z3-4",
    "km_z5_t1_t2": "km Z5-T1-T2",
    "km_sprinting": "km sprinting",
    "strength_training": "strength training",
    "hours_alternative": "hours alternative",
    "perceived_exertion": "perceived exertion",
    "perceived_training_success": "perceived trainingSuccess",
    "perceived_recovery": "perceived recovery",
}
# OPEN ISSUE: the research data stores perceived scores on 0-1 (0.01 steps), but its
# documentation we have does not say which direction "perceived recovery" runs.
# Within athletes it rises after hard days (r = +0.37 with the previous day's
# exertion), which hints that higher = LESS recovered, but all three scores also
# move together (response style). The app currently asks "How recovered do you
# feel?" (higher = better). Confirm against the methods of Lövdal et al. (2021),
# doi:10.1123/ijspp.2020-0518, before relying on recovery-driven predictions.
PERCEIVED_FIELDS = ["perceived_exertion", "perceived_training_success", "perceived_recovery"]
WINDOW_DAYS = 21  # three weekly totals are the longest look-back the model uses


def reconstruct_daily_logs(day_df, athlete):
    """Rebuild one research athlete's day-by-day diary from the overlapping 7-day windows.

    Row ``d`` of the day file holds days d-7 (suffix "") ... d-1 (suffix ".6"); every
    calendar day appears in up to 7 rows and is taken from whichever row contains it.
    Used by the consistency tests and to load demo athletes into the app.
    Returns a DataFrame with an integer ``date`` column and LOG_FIELDS columns.
    """
    rows = day_df[day_df["Athlete ID"] == athlete]
    days = {}
    for _, r in rows.iterrows():
        for k in range(7):
            date = int(r["Date"]) - 7 + k
            days[date] = {f: r[data._day_col(col, k)] for f, col in LOG_FIELDS.items()}
    logs = pd.DataFrame.from_dict(days, orient="index").rename_axis("date").reset_index()
    for f in PERCEIVED_FIELDS:
        logs.loc[logs[f] < 0, f] = np.nan  # -0.01 = rest-day sentinel
    return logs.sort_values("date").reset_index(drop=True)


def daily_calendar(logs, start=None, end=None):
    """One row per calendar day with every LOG_FIELDS column; unlogged days become rest days.

    ``logs`` is a DataFrame with a ``date`` column (datetime-like or integer day
    numbers) and any subset of LOG_FIELDS columns.
    """
    logs = logs.copy()
    idx_kind_is_int = np.issubdtype(np.asarray(logs["date"]).dtype, np.integer)
    if not idx_kind_is_int:
        logs["date"] = pd.to_datetime(logs["date"]).dt.normalize()
    logs = logs.drop_duplicates("date", keep="last").set_index("date").sort_index()

    start = logs.index.min() if start is None else start
    end = logs.index.max() if end is None else end
    calendar = (pd.RangeIndex(start, end + 1) if idx_kind_is_int else pd.date_range(start, end, freq="D"))
    cal = logs.reindex(calendar)
    cal["logged"] = cal.index.isin(logs.index)

    for field in LOG_FIELDS:
        if field not in cal:
            cal[field] = np.nan
        if field not in PERCEIVED_FIELDS:
            # Volume fields: an unlogged day is a rest day (zero training)
            cal[field] = cal[field].fillna(0.0)
        # Perceived scores stay NaN on rest days, exactly like the dataset's -0.01 sentinel
        # (which data.runner_engineered_features also turns into NaN).
    cal.index.name = "date"
    return cal


def wide_windows(cal):
    """Reshape a daily calendar into the dataset's wide layout, one row per prediction day.

    Row for day D holds day columns for D-7 (suffix "") ... D-1 (suffix ".6") and
    weekly km totals for D-7..D-1 ("total kms"), D-14..D-8 (".1") and D-21..D-15 (".2").
    The first prediction day is the day after the first log; the last is the day
    after the last log ("tomorrow").
    """
    step = 1 if isinstance(cal.index, pd.RangeIndex) else pd.Timedelta(days=1)
    # Extend the calendar by one day so the day after the last log gets a row
    cal = cal.reindex(cal.index.append(pd.Index([cal.index[-1] + step]))) if len(cal) else cal
    out = {}
    for field, col in LOG_FIELDS.items():
        series = cal[field]
        for d in range(7):
            # shift(7 - d): the value from (7 - d) days before the prediction day
            out[data._day_col(col, d)] = series.shift(7 - d)
    km = cal["total_km"].fillna(0.0)
    for w in range(3):
        # Rolling 7-day sum ending w weeks before yesterday
        out[data._week_col("total kms", w)] = km.shift(1 + 7 * w).rolling(7, min_periods=7).sum()
    logged = cal["logged"].astype(float).fillna(0.0)
    wide = pd.DataFrame(out, index=cal.index)
    wide["days_logged_last_21"] = logged.shift(1).rolling(WINDOW_DAYS, min_periods=1).sum()
    return wide.iloc[1:]  # the first calendar day has no history at all


def features_from_logs(logs, athlete_id="athlete", feature_names=None, history_end=None):
    """Model features for every prediction day of one athlete.

    ``history_end`` extends the calendar to that day (unlogged days count as rest),
    so the last returned row is the risk for ``history_end + 1``. Use it to get
    "risk for today" (history_end = yesterday) when the last log is older.

    Returns (features, info): ``features`` has exactly ``feature_names`` columns
    (the trained model's order) and ``info`` holds ``days_logged_last_21`` and
    ``has_full_history`` (21 days of history available).
    """
    # Dates may be integer day numbers (research data, tests) or real calendar dates (API)
    is_int = np.issubdtype(np.asarray(logs["date"]).dtype, np.integer)
    last_log = logs["date"].max() if is_int else pd.to_datetime(logs["date"]).max().normalize()
    if history_end is not None and not is_int:
        history_end = pd.Timestamp(history_end).normalize()
    end = last_log if history_end is None else max(last_log, history_end)
    cal = daily_calendar(logs, end=end)
    wide = wide_windows(cal)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-rest weeks -> NaN means, as in training
        eng = data.runner_engineered_features(wide)
    athlete = pd.Series(athlete_id, index=wide.index)
    date = pd.Series(np.arange(len(wide)), index=wide.index)  # row order = calendar order
    personal = data.personal_baseline_features(eng, athlete, date)
    feats = pd.concat([eng, personal], axis=1)

    info = pd.DataFrame({
        "days_logged_last_21": wide["days_logged_last_21"],
        # Weekly totals need 21 days; earlier rows are computed but less reliable
        "has_full_history": wide[data._week_col("total kms", 2)].notna(),
    }, index=wide.index)
    if feature_names is not None:
        feats = feats[list(feature_names)]
    return feats, info
