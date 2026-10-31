"""Request/response shapes. Validation ranges stop impossible values reaching the model."""
import datetime as dt

from pydantic import BaseModel, Field, model_validator


class AthleteCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    sport: str = "Running"


class AthleteOut(BaseModel):
    id: int
    name: str
    sport: str


class LogIn(BaseModel):
    """A day's training. Upper bounds are generous but physically plausible."""
    sessions: int = Field(0, ge=0, le=5)
    total_km: float = Field(0.0, ge=0, le=150)
    km_z3_4: float = Field(0.0, ge=0, le=150)
    km_z5_t1_t2: float = Field(0.0, ge=0, le=150)
    km_sprinting: float = Field(0.0, ge=0, le=50)
    strength_training: float = Field(0.0, ge=0, le=5)
    hours_alternative: float = Field(0.0, ge=0, le=12)
    perceived_exertion: float | None = Field(None, ge=0, le=1)
    perceived_training_success: float | None = Field(None, ge=0, le=1)
    perceived_recovery: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def zones_within_total(self):
        # Intensity zones are parts of the total distance, so they cannot exceed it
        if self.km_z3_4 + self.km_z5_t1_t2 + self.km_sprinting > self.total_km + 1e-9:
            raise ValueError("km in zones (Z3-4 + Z5/T1/T2 + sprinting) cannot exceed total_km")
        return self


class LogOut(LogIn):
    date: dt.date


class Reason(BaseModel):
    feature: str
    label: str
    value: float | None
    contribution: float  # effect on the log-odds of injury; positive raises risk


class RiskPoint(BaseModel):
    date: dt.date
    probability: float          # calibrated probability of an injury on this day
    relative_risk: float        # probability / average day's probability
    flagged: bool               # above the alert threshold
    days_logged_last_21: int
    has_full_history: bool      # 21 days of history available
    baseline_ready: bool        # enough history for personal-baseline features


class RiskOut(BaseModel):
    athlete: AthleteOut
    timeline: list[RiskPoint]
    latest: RiskPoint | None
    reasons_up: list[Reason]    # factors raising the latest risk
    reasons_down: list[Reason]  # factors lowering it


class TeamRow(BaseModel):
    athlete: AthleteOut
    risk: RiskPoint | None
    top_reason: Reason | None
    last_log_date: dt.date | None


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    n_athletes: int
    cv_roc_auc: float
    threshold: float
    base_rate: float
