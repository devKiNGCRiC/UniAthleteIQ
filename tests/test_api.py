"""End-to-end API tests against a throw-away SQLite database."""
import datetime as dt

import pytest

from uniathleteiq import paths

MODEL = paths.MODELS_DIR / "production" / "runner_model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/train_production_model.py first")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlmodel import create_engine

    import api.db
    import api.main
    # Swap in a throw-away SQLite engine; get_session/create_db_and_tables read the
    # module-level `engine` at call time, so every request in this test uses it.
    test_engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                                connect_args={"check_same_thread": False})
    monkeypatch.setattr(api.db, "engine", test_engine)
    with TestClient(api.main.app) as c:
        yield c


def _log(km=10.0, exertion=0.5):
    return {"sessions": 1, "total_km": km, "km_z3_4": 2.0, "perceived_exertion": exertion,
            "perceived_training_success": 0.6, "perceived_recovery": 0.5}


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_full_flow(client):
    a = client.post("/api/athletes", json={"name": "Test Runner"}).json()
    today = dt.date.today()
    for i in range(40, 0, -1):  # 40 days of logs ending yesterday; every 3rd day is a rest day
        day = (today - dt.timedelta(days=i)).isoformat()
        body = _log() if i % 3 else {}
        assert client.put(f"/api/athletes/{a['id']}/logs/{day}", json=body).status_code == 200

    r = client.get(f"/api/athletes/{a['id']}/risk?days=30").json()
    assert len(r["timeline"]) == 30
    assert r["latest"]["date"] == today.isoformat()  # history to yesterday -> risk for today
    assert r["latest"]["days_logged_last_21"] == 21
    assert r["latest"]["has_full_history"] and r["latest"]["baseline_ready"]
    assert 0 < r["latest"]["probability"] < 1
    assert r["reasons_up"] or r["reasons_down"]

    team = client.get("/api/team/today").json()
    assert team[0]["athlete"]["id"] == a["id"] and team[0]["risk"] is not None


def test_validation_rejects_impossible_logs(client):
    a = client.post("/api/athletes", json={"name": "V"}).json()
    day = dt.date.today().isoformat()
    # zones exceed total distance
    assert client.put(f"/api/athletes/{a['id']}/logs/{day}",
                      json={"total_km": 5, "km_z3_4": 10}).status_code == 422
    # perceived scores are on a 0-1 scale
    assert client.put(f"/api/athletes/{a['id']}/logs/{day}", json={"perceived_exertion": 7}).status_code == 422
    # no logging the future
    future = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    assert client.put(f"/api/athletes/{a['id']}/logs/{future}", json={}).status_code == 422


def test_upsert_overwrites_same_day(client):
    a = client.post("/api/athletes", json={"name": "U"}).json()
    day = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    client.put(f"/api/athletes/{a['id']}/logs/{day}", json=_log(km=5))
    client.put(f"/api/athletes/{a['id']}/logs/{day}", json=_log(km=12))
    logs = client.get(f"/api/athletes/{a['id']}/logs").json()
    assert len(logs) == 1 and logs[0]["total_km"] == 12


def test_csv_import_is_all_or_nothing(client):
    a = client.post("/api/athletes", json={"name": "C"}).json()
    good = "date,sessions,total_km,perceived_exertion\n2026-01-01,1,8,0.4\n2026-01-02,0,0,\n"
    r = client.post(f"/api/athletes/{a['id']}/logs/import", files={"file": ("l.csv", good, "text/csv")})
    assert r.json() == {"imported": 2}
    bad = "date,total_km\n2026-01-03,9\n2026-01-04,-5\n"  # second row invalid
    assert client.post(f"/api/athletes/{a['id']}/logs/import",
                       files={"file": ("l.csv", bad, "text/csv")}).status_code == 422
    assert len(client.get(f"/api/athletes/{a['id']}/logs").json()) == 2  # nothing from the bad file


def test_unknown_athlete_404(client):
    assert client.get("/api/athletes/999/risk").status_code == 404
