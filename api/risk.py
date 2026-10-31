"""Risk service: loads the trained model once and scores athletes from their logs."""
import datetime as dt
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from uniathleteiq import data, explain, serving

from . import config, schemas


class RiskModel:
    def __init__(self, path=config.MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found at {path}. Run scripts/train_production_model.py")
        b = joblib.load(path)
        self.model = b["model"]
        self.features = b["features"]
        self.threshold = b["threshold"]
        self.platt = b["platt"]
        self.base_rate = b["base_rate"]
        self.metadata = b["metadata"]

    def _calibrate(self, score):
        p = np.clip(score, 1e-6, 1 - 1e-6)
        return self.platt.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]

    def score_logs(self, logs: pd.DataFrame, history_end: dt.date | None = None):
        """Risk for every day from the day after the first log to ``history_end + 1``.

        Returns (timeline DataFrame, features DataFrame) indexed by date.
        """
        X, info = serving.features_from_logs(logs, feature_names=self.features, history_end=history_end)
        score = self.model.predict_proba(X)[:, 1]
        prob = self._calibrate(score)
        timeline = pd.DataFrame({
            "date": X.index.date,
            "probability": prob,
            "relative_risk": prob / self.base_rate,
            "flagged": score >= self.threshold,
            "days_logged_last_21": info["days_logged_last_21"].astype(int).to_numpy(),
            "has_full_history": info["has_full_history"].to_numpy(),
            # Personal z-scores exist only after PERSONAL_MIN_HISTORY rows of history
            "baseline_ready": np.arange(len(X)) >= data.PERSONAL_MIN_HISTORY,
        }, index=X.index)
        return timeline, X

    def reasons(self, X_row: pd.DataFrame, k_up=4, k_down=3):
        """Top factors raising and lowering risk for a single day (exact for a linear model)."""
        contrib = explain.linear_contributions(self.model, X_row).iloc[0].sort_values()
        def make(f):
            v = X_row.iloc[0][f]
            return schemas.Reason(feature=f, label=explain.feature_label(f),
                                  value=None if pd.isna(v) else float(v), contribution=float(contrib[f]))
        up = [make(f) for f in contrib.index[::-1][:k_up] if contrib[f] > 0]
        down = [make(f) for f in contrib.index[:k_down] if contrib[f] < 0]
        return up, down


@lru_cache(maxsize=1)
def get_risk_model() -> RiskModel:
    """Loaded lazily and cached: the bundle is read once per process."""
    return RiskModel()


def logs_to_frame(logs) -> pd.DataFrame:
    """DailyLog rows -> the DataFrame layout serving.features_from_logs expects."""
    cols = ["date", *serving.LOG_FIELDS]
    return pd.DataFrame([{c: getattr(log, c) for c in cols} for log in logs], columns=cols)
