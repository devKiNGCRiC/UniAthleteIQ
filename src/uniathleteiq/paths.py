from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "datasets"
OUTPUT_DIR = BASE / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts" / "v2"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

RUNNERS_DAY_CSV = DATA_DIR / "day_approach_maskedID_timeseries.csv"
RUNNERS_WEEK_CSV = DATA_DIR / "week_approach_maskedID_timeseries.csv"
FOOTBALL_CSV = DATA_DIR / "dataset.csv"
MULTISPORT_CSV = DATA_DIR / "sports_injury_detection_dataset.csv"


def ensure_output_dirs():
    for d in (CHARTS_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
