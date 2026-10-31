"""FastAPI application.

Run locally:
    uvicorn api.main:app --reload
Interactive docs at http://localhost:8000/docs
"""
import datetime as dt
import io
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlmodel import Session, delete, select

from uniathleteiq import serving

from . import config, schemas
from .db import Athlete, DailyLog, create_db_and_tables, get_session
from .risk import get_risk_model, logs_to_frame

DISCLAIMER = ("Decision-support only, not medical advice. The model ranks days by injury risk with "
              "modest accuracy; most flagged days are not followed by an injury, and some injuries are missed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: make sure tables exist and fail fast if the model bundle is missing
    create_db_and_tables()
    get_risk_model()
    yield


app = FastAPI(title="UniAthleteIQ API", version="1.0.0", description=DISCLAIMER, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

SessionDep = Annotated[Session, Depends(get_session)]


def _athlete_or_404(session: Session, athlete_id: int) -> Athlete:
    athlete = session.get(Athlete, athlete_id)
    if not athlete:
        raise HTTPException(404, "Athlete not found")
    return athlete


def _athlete_out(a: Athlete) -> schemas.AthleteOut:
    return schemas.AthleteOut(id=a.id, name=a.name, sport=a.sport)


def _athlete_logs(session: Session, athlete_id: int):
    return session.exec(select(DailyLog).where(DailyLog.athlete_id == athlete_id).order_by(DailyLog.date)).all()


# ── Meta ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_version": get_risk_model().metadata["version"]}


@app.get("/api/model", response_model=schemas.ModelInfo)
def model_info():
    m = get_risk_model()
    return schemas.ModelInfo(version=m.metadata["version"], trained_at=m.metadata["trained_at"],
                             n_athletes=m.metadata["n_athletes"], cv_roc_auc=m.metadata["cv_roc_auc"],
                             threshold=m.threshold, base_rate=m.base_rate)


# ── Athletes ────────────────────────────────────────────────────────────────

@app.get("/api/athletes", response_model=list[schemas.AthleteOut])
def list_athletes(session: SessionDep):
    return [_athlete_out(a) for a in session.exec(select(Athlete).order_by(Athlete.name)).all()]


@app.post("/api/athletes", response_model=schemas.AthleteOut, status_code=201)
def create_athlete(body: schemas.AthleteCreate, session: SessionDep):
    athlete = Athlete(name=body.name.strip(), sport=body.sport)
    session.add(athlete)
    session.commit()
    session.refresh(athlete)
    return _athlete_out(athlete)


@app.get("/api/athletes/{athlete_id}", response_model=schemas.AthleteOut)
def get_athlete(athlete_id: int, session: SessionDep):
    return _athlete_out(_athlete_or_404(session, athlete_id))


@app.delete("/api/athletes/{athlete_id}", status_code=204)
def delete_athlete(athlete_id: int, session: SessionDep):
    athlete = _athlete_or_404(session, athlete_id)
    session.exec(delete(DailyLog).where(DailyLog.athlete_id == athlete_id))
    session.delete(athlete)
    session.commit()


# ── Daily logs ──────────────────────────────────────────────────────────────

@app.get("/api/athletes/{athlete_id}/logs", response_model=list[schemas.LogOut])
def list_logs(athlete_id: int, session: SessionDep,
              start: dt.date | None = None, end: dt.date | None = None):
    _athlete_or_404(session, athlete_id)
    q = select(DailyLog).where(DailyLog.athlete_id == athlete_id)
    if start:
        q = q.where(DailyLog.date >= start)
    if end:
        q = q.where(DailyLog.date <= end)
    return [schemas.LogOut(**log.model_dump()) for log in session.exec(q.order_by(DailyLog.date)).all()]


def _upsert_log(session: Session, athlete_id: int, day: dt.date, body: schemas.LogIn) -> DailyLog:
    """Insert the day's log, or overwrite it if the athlete already logged that day."""
    log = session.exec(select(DailyLog).where(DailyLog.athlete_id == athlete_id,
                                              DailyLog.date == day)).first()
    if log is None:
        log = DailyLog(athlete_id=athlete_id, date=day)
    log.sqlmodel_update(body.model_dump())
    session.add(log)
    return log


@app.put("/api/athletes/{athlete_id}/logs/{day}", response_model=schemas.LogOut)
def put_log(athlete_id: int, day: dt.date, body: schemas.LogIn, session: SessionDep):
    _athlete_or_404(session, athlete_id)
    if day > dt.date.today():
        raise HTTPException(422, "Cannot log training for a future date")
    log = _upsert_log(session, athlete_id, day, body)
    session.commit()
    session.refresh(log)
    return schemas.LogOut(**log.model_dump())


@app.delete("/api/athletes/{athlete_id}/logs/{day}", status_code=204)
def delete_log(athlete_id: int, day: dt.date, session: SessionDep):
    session.exec(delete(DailyLog).where(DailyLog.athlete_id == athlete_id, DailyLog.date == day))
    session.commit()


@app.post("/api/athletes/{athlete_id}/logs/import")
async def import_logs(athlete_id: int, session: SessionDep, file: UploadFile = File(...)):
    """Bulk import from CSV: a ``date`` column plus any of the log fields (see /docs).

    The whole file is validated first; nothing is saved if any row is invalid.
    """
    _athlete_or_404(session, athlete_id)
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:  # malformed CSV
        raise HTTPException(422, f"Could not read CSV: {exc}")
    if "date" not in df.columns:
        raise HTTPException(422, "CSV needs a 'date' column (YYYY-MM-DD)")
    unknown = set(df.columns) - {"date", *serving.LOG_FIELDS}
    if unknown:
        raise HTTPException(422, f"Unknown columns: {sorted(unknown)}")

    rows, errors = [], []
    for i, rec in enumerate(df.to_dict("records"), start=2):  # line 1 is the header
        try:
            day = pd.Timestamp(rec.pop("date")).date()
            body = schemas.LogIn(**{k: (None if pd.isna(v) else v) for k, v in rec.items()})
            rows.append((day, body))
        except (ValidationError, ValueError) as exc:
            errors.append(f"line {i}: {exc}")
    if errors:
        raise HTTPException(422, {"errors": errors[:20], "n_errors": len(errors)})
    for day, body in rows:
        _upsert_log(session, athlete_id, day, body)
    session.commit()
    return {"imported": len(rows)}


# ── Risk ────────────────────────────────────────────────────────────────────

def _risk_point(row) -> schemas.RiskPoint:
    return schemas.RiskPoint(**{k: row[k] for k in schemas.RiskPoint.model_fields})


@app.get("/api/athletes/{athlete_id}/risk", response_model=schemas.RiskOut)
def athlete_risk(athlete_id: int, session: SessionDep, days: int = Query(90, ge=7, le=730)):
    """Risk timeline for the last ``days`` days up to *today*, with reasons for the latest day."""
    athlete = _athlete_or_404(session, athlete_id)
    logs = _athlete_logs(session, athlete_id)
    if not logs:
        return schemas.RiskOut(athlete=_athlete_out(athlete), timeline=[], latest=None,
                               reasons_up=[], reasons_down=[])
    m = get_risk_model()
    # history up to yesterday -> the last row is today's risk (or tomorrow's if today is logged)
    history_end = dt.date.today() - dt.timedelta(days=1)
    timeline, X = m.score_logs(logs_to_frame(logs), history_end=history_end)
    timeline = timeline.iloc[-days:]
    up, down = m.reasons(X.iloc[[-1]])
    points = [_risk_point(r) for _, r in timeline.iterrows()]
    return schemas.RiskOut(athlete=_athlete_out(athlete), timeline=points, latest=points[-1],
                           reasons_up=up, reasons_down=down)


@app.get("/api/team/today", response_model=list[schemas.TeamRow])
def team_today(session: SessionDep):
    """Every athlete's latest risk, highest first: the coach's morning view."""
    m = get_risk_model()
    history_end = dt.date.today() - dt.timedelta(days=1)
    rows = []
    for athlete in session.exec(select(Athlete)).all():
        logs = _athlete_logs(session, athlete.id)
        if not logs:
            rows.append(schemas.TeamRow(athlete=_athlete_out(athlete), risk=None, top_reason=None,
                                        last_log_date=None))
            continue
        timeline, X = m.score_logs(logs_to_frame(logs), history_end=history_end)
        up, _ = m.reasons(X.iloc[[-1]], k_up=1, k_down=0)
        rows.append(schemas.TeamRow(athlete=_athlete_out(athlete), risk=_risk_point(timeline.iloc[-1]),
                                    top_reason=up[0] if up else None, last_log_date=logs[-1].date))
    # Highest relative risk first; athletes without logs at the end
    return sorted(rows, key=lambda r: -(r.risk.relative_risk if r.risk else -1))
