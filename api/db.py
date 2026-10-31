"""Database tables and session handling (SQLModel = SQLAlchemy + Pydantic)."""
import datetime as dt

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine

from . import config


class Athlete(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    sport: str = "Running"  # the current model is trained on runners only
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class DailyLog(SQLModel, table=True):
    """One day of an athlete's training diary. Field meanings follow the research dataset.

    Perceived scores use the dataset's 0-1 scale (the frontend shows 0-10 sliders);
    they are left empty on rest days.
    """
    __table_args__ = (UniqueConstraint("athlete_id", "date"),)  # one log per athlete per day

    id: int | None = Field(default=None, primary_key=True)
    athlete_id: int = Field(foreign_key="athlete.id", index=True)
    date: dt.date = Field(index=True)
    sessions: int = 0
    total_km: float = 0.0
    km_z3_4: float = 0.0          # threshold / tempo km
    km_z5_t1_t2: float = 0.0      # km at or above race pace (Z5 and interval types T1/T2)
    km_sprinting: float = 0.0
    strength_training: float = 0.0  # number of strength sessions
    hours_alternative: float = 0.0  # cross-training hours (bike, swim, ...)
    perceived_exertion: float | None = None
    perceived_training_success: float | None = None
    perceived_recovery: float | None = None


# check_same_thread=False lets FastAPI's worker threads share a SQLite connection;
# other databases (Postgres) do not accept the option.
_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)


def create_db_and_tables():
    if config.DATABASE_URL.startswith("sqlite"):
        config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency: one session per request, closed afterwards."""
    with Session(engine) as session:
        yield session
