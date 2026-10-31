"""Per-prediction explanations for the linear runner model.

For a standardised logistic regression, coefficient x scaled value is exactly the
SHAP value under an independent (mean) background, so no sampling is needed.
"""
import numpy as np
import pandas as pd

# Plain-language names for the runner model's features, shared by the API and dashboards
FEATURE_LABELS = {
    "km_7d": "Total km, last 7 days",
    "km_last_3d": "Total km, last 3 days",
    "km_yesterday": "Km yesterday",
    "hard_km_7d": "Hard km (Z5/intervals/sprint), 7 days",
    "rest_days_7d": "Rest days, last 7 days",
    "sessions_7d": "Sessions, last 7 days",
    "monotony_7d": "Training monotony (mean/SD of daily km)",
    "strain_7d": "Training strain (load x monotony)",
    "acwr_3wk": "Acute:chronic workload ratio",
    "km_change_vs_prev_week": "Km change vs previous week",
    "km_ratio_week0_vs_1": "This week's km vs last week's",
    "km_ratio_week0_vs_2": "This week's km vs 2 weeks ago",
    "km_ratio_week1_vs_2": "Last week's km vs 2 weeks ago",
    "exertion_mean_7d": "Avg perceived exertion",
    "exertion_max_7d": "Max perceived exertion",
    "exertion_trend": "Exertion trend (recent vs earlier days)",
    "recovery_mean_7d": "Avg perceived recovery",
    "recovery_min_7d": "Worst perceived recovery",
    "recovery_trend": "Recovery trend (recent vs earlier days)",
    "training_success_mean_7d": "Avg perceived training success",
    "exertion_minus_recovery_7d": "Exertion minus recovery",
}


def feature_label(name):
    """Human-readable label; personal-baseline features say they are relative to the athlete."""
    if name.endswith("_vs_personal"):
        base = name.removesuffix("_vs_personal")
        return f"{FEATURE_LABELS.get(base, base)} (vs athlete's normal)"
    return FEATURE_LABELS.get(name, name)


def linear_contributions(pipeline, X):
    """Contribution of each feature to the log-odds of injury, one row per sample."""
    pre = pipeline[:-1]
    clf = pipeline[-1]
    Z = pre.transform(X)
    names = pre.get_feature_names_out()
    contrib = pd.DataFrame(Z * clf.coef_[0], columns=names, index=X.index)
    # Fold "missingindicator_<f>" columns into their parent feature
    out = pd.DataFrame(index=X.index)
    for n in names:
        parent = n.removeprefix("missingindicator_")
        out[parent] = out.get(parent, 0) + contrib[n]
    return out


def top_reasons(contrib_row, X_row, k=5):
    """The k features pushing risk up the most for one athlete-day."""
    s = contrib_row.sort_values(ascending=False).head(k)
    return pd.DataFrame({"feature": s.index, "value": [X_row.get(f, np.nan) for f in s.index],
                         "log_odds_contribution": s.values})
