"""Runtime settings, read from environment variables so the same code runs locally and in the cloud."""
import os
from pathlib import Path

from uniathleteiq import paths

INSTANCE_DIR = paths.BASE / "instance"  # local, git-ignored storage (SQLite file)

# SQLite locally; set DATABASE_URL to a Postgres URL (e.g. Supabase) in production.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(INSTANCE_DIR / 'uniathleteiq.db').as_posix()}")

# Model bundle produced by scripts/train_production_model.py
MODEL_PATH = Path(os.environ.get("MODEL_PATH", paths.MODELS_DIR / "production" / "runner_model.joblib"))

# Browser origins allowed to call the API (the Next.js frontend). Comma-separated.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
