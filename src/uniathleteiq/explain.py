"""Per-prediction explanations for the linear runner model.

For a standardised logistic regression, coefficient x scaled value is exactly the
SHAP value under an independent (mean) background, so no sampling is needed.
"""
import numpy as np
import pandas as pd


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
