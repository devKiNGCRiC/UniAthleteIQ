"""Load a few real athletes from the research dataset into the app as demo data.

The research data uses anonymous day numbers, not dates. Each athlete's diary
is shifted so that their last recorded day becomes *yesterday*, which makes the
dashboard look like a live team. Names are clearly marked as demo athletes.

Note: the production model was trained on all research athletes, including
these, so their risk curves look somewhat better than a brand-new athlete's
would. Use them to explore the product, not to judge accuracy (see notebook 02/03).

Usage:
    python scripts/seed_demo.py              # 6 demo athletes, last 180 days each
    python scripts/seed_demo.py --reset      # delete existing demo athletes first
"""
import argparse
import datetime as dt

import pandas as pd
from sqlmodel import Session, delete, select

from api.db import Athlete, DailyLog, create_db_and_tables, engine
from uniathleteiq import paths, serving

DEMO_PREFIX = "Demo runner"


def pick_athletes(day_df, n):
    """Athletes with long records and several injuries make the most informative demos."""
    stats = day_df.groupby("Athlete ID").agg(days=("Date", "size"), injuries=("injury", "sum"))
    return stats[(stats.days > 400)].sort_values("injuries", ascending=False).head(n).index.tolist()


def densest_recent_window(logs, days, min_coverage=0.95):
    """The latest ``days``-long stretch with at least ``min_coverage`` of days recorded.

    The research records contain gaps (e.g. periods excluded around injuries).
    A demo athlete with a long gap at the end would show "not logged" warnings,
    so we pick a well-covered window instead. Falls back to the best-covered one.
    """
    dates = set(logs["date"].astype(int))
    best_end, best_cov = None, -1.0
    for end in sorted(dates, reverse=True):
        cov = sum(d in dates for d in range(end - days + 1, end + 1)) / days
        if cov >= min_coverage:
            best_end = end
            break
        if cov > best_cov:
            best_end, best_cov = end, cov
    return logs[(logs["date"] > best_end - days) & (logs["date"] <= best_end)]


def main(n, days, reset):
    create_db_and_tables()
    day_df = pd.read_csv(paths.RUNNERS_DAY_CSV)
    yesterday = dt.date.today() - dt.timedelta(days=1)

    with Session(engine) as session:
        if reset:
            demo = session.exec(select(Athlete).where(Athlete.name.startswith(DEMO_PREFIX))).all()
            for a in demo:
                session.exec(delete(DailyLog).where(DailyLog.athlete_id == a.id))
                session.delete(a)
            session.commit()

        for i, research_id in enumerate(pick_athletes(day_df, n), start=1):
            logs = densest_recent_window(serving.reconstruct_daily_logs(day_df, research_id), days)
            last = int(logs["date"].max())
            athlete = Athlete(name=f"{DEMO_PREFIX} {i} (research #{research_id})", sport="Running")
            session.add(athlete)
            session.flush()  # assigns athlete.id
            for rec in logs.to_dict("records"):
                day = yesterday - dt.timedelta(days=last - int(rec.pop("date")))
                clean = {k: (None if pd.isna(v) else float(v)) for k, v in rec.items()}
                clean["sessions"] = int(clean["sessions"] or 0)
                session.add(DailyLog(athlete_id=athlete.id, date=day, **clean))
            print(f"  {athlete.name}: {len(logs)} days")
        session.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="number of demo athletes")
    ap.add_argument("--days", type=int, default=180, help="days of history per athlete")
    ap.add_argument("--reset", action="store_true", help="remove existing demo athletes first")
    a = ap.parse_args()
    main(a.n, a.days, a.reset)
