# 🧠 UniAthleteIQ — AI-Powered Athlete Injury Risk Prediction

## 🚀 Overview

UniAthleteIQ predicts athlete injury risk from training-load and wellness logs, explains each prediction to coaches, and ships as a working web app.

It has two parts:

* **Research:** leakage-free models on public datasets (competitive running, football, multi-sport wearables), with statistical validation (repeated athlete-grouped CV, athlete-bootstrap confidence intervals, paired tests, baselines and sensitivity analyses) and an explicit cross-sport test.
* **Product:** a FastAPI backend and a Next.js web app where athletes log training in about 30 seconds and coaches see who to check in with today, and why.

See [What changed in v2](#-what-changed-in-v2) for why the original v1 numbers (96% accuracy) were not reliable.

---

## 🎯 Problem Statement

* Sports injuries are often detected **after they occur**
* Most ML models are **sport-specific**
* Coaches lack **interpretable insights** from predictions

---

## 💡 Approach

* 🔮 **Early warning:** rank each athlete-day by injury risk from the previous 1–3 weeks of training
* 🧍 **Personal baselines:** compare today's load and wellness with *that athlete's own normal*
* 🌍 **Cross-sport test:** do load and effort signals learned on one sport transfer to another?
* 🧠 **Explainable AI:** exact per-prediction explanations (SHAP)
* 🔒 **Leakage-free evaluation:** athlete-grouped CV, held-out athletes, and statistical tests

---

## 📊 Datasets (Kaggle, public)

| Dataset | Unit | Size | Used for |
|---|---|---|---|
| Competitive runners: daily + weekly training logs | athlete-day with 7-day / 3-week history | 42,680 days, 74 athletes, 575 injuries | **Main risk model** |
| EPL football players | player-season | 1,301 seasons, 604 players | Severe-injury season (more than 28 days out) |
| Multi-sport wearable sessions | session | 1,000 sessions | Secondary; appears synthetic (see notebook 02 §5) |
| Basketball biomechanics, NBA injury list | injured players only | — | Not usable for risk models (no uninjured comparison group) |

---

## ⚙️ Methodology

1. **Feature engineering (runners):** 7-day load, hard (Z5, interval, sprint) km, rest days, **ACWR** (acute:chronic workload ratio), **monotony** and **strain** (Foster), week-over-week load log-ratios, and perceived exertion, recovery and training-success means and trends.
2. **Personal baselines:** each feature as a z-score against the athlete's *own past days only*. There is no look-ahead, and a unit test enforces this.
3. **Imbalance handling:** class weights. SMOTE is only ever applied inside training folds; it gave no benefit.
4. **Models:** Logistic Regression ⭐ (final), Random Forest, XGBoost.
5. **Evaluation:** 10× repeated 5-fold **athlete-grouped** CV; 95% CIs from an **athlete-level bootstrap**; **paired** bootstrap tests with Holm correction; untrained baselines (ACWR, the "ACWR > 1.5" rule, 7-day km, perceived exertion); sensitivity analyses; a final test on 15 athletes held out from the start. Main metrics are ROC-AUC and PR-AUC on the real class balance (1.35% of runner-days are injuries).
6. **Explainability and calibration:** exact SHAP values for the linear model, Platt-calibrated probabilities, and an alert threshold chosen on training data only.

---

## 🏆 Results

### Runners (main model)

| Evaluation | ROC-AUC (95% CI) | PR-AUC (random guess) |
|---|---|---|
| 10× repeated athlete-grouped CV, 59 athletes | **0.655** (0.613–0.696) | 0.026 (0.014) |
| 15 held-out athletes, tested once | 0.620 (0.522–0.698) | 0.020 (0.012) |
| Same athletes, last 20% of each timeline | 0.623 | 0.033 (0.019) |

**In practice:** flagging the riskiest 20% of runner-days catches about **32% of injuries**, compared with 20% for random flagging. It's useful for prioritising coach check-ins, not for diagnosing individual athletes.

### What the statistics show ([notebook 03](notebooks/03_statistical_validation.ipynb))

* **Personal baselines significantly improve every model:** LR +0.024 ROC-AUC (95% CI 0.003–0.047), RF +0.057, XGBoost +0.048, all significant after Holm correction.
* **Model choice does not matter:** LR, RF and XGBoost are statistically indistinguishable, so the simplest, fully explainable model is used.
* **The model beats common workload rules:** +0.11 over ACWR, +0.12 over 7-day km and +0.16 over the popular **"ACWR > 1.5" rule, which performs at chance** in this data (ROC-AUC 0.49).
* **Perceived effort carries most of the signal.** The athlete's average perceived exertion alone reaches 0.63, and the full model's advantage over it (+0.03) is not significant. Asking athletes how hard training felt is the most valuable thing to collect.
* **Robust:** results barely change with the personal-history window (14, 28 or 56 days) or the football severity cut-off (14, 28 or 56 days). Most of the uncertainty comes from *which athletes* are in the data, not from how the CV folds are split.

### Other datasets

| Experiment | Evaluated on | ROC-AUC | PR-AUC (random guess) |
|---|---|---|---|
| Football: severe-injury season | held-out players | 0.652 | 0.799 (0.708) |
| Multi-sport wearable | 5-fold CV | 0.596 | 0.697 (0.617) |
| Cross-sport: train runners → test multi-sport | other dataset | 0.593 | — |
| Cross-sport: train multi-sport → test runners | other dataset | 0.610 | — |

With dataset-normalised load and effort, models transfer across sources about as well as within a source (≈ 0.59–0.61). The signal is weak, and the multi-sport data appears synthetic.

Charts: [outputs/charts/v2/](outputs/charts/v2/), [outputs/charts/v3/](outputs/charts/v3/). Tables: [outputs/results/](outputs/results/).

---

## 💻 The web app

| Screen | Purpose |
|---|---|
| **Team** | Everyone ranked by today's risk on a "lane" track (1× = an average day), with status and the main factor |
| **Athlete** | Today's risk, risk over 30/90/180 days, factors raising and lowering risk, last two weeks of training |
| **Log training** | Phone-friendly daily form: sessions, km, optional intensity breakdown, effort / success / recovery sliders |
| **How it works** | Plain-language explanation with live accuracy figures from the model |

The API reuses the research feature code, and a test checks that features computed from daily logs **exactly match** the features the model was trained on (no training/serving skew). Data-quality warnings appear when fewer than 18 of the last 21 days are logged or the personal baseline is still forming.

---

## 🔍 What changed in v2

The v1 notebook reported 96% accuracy (Random Forest). An audit ([notebook 02 §1](notebooks/02_leakage_free_pipeline.ipynb)) showed this came from data leakage:

| v1 problem | Effect |
|---|---|
| SMOTE applied **before** the train/test split | synthetic near-copies of test rows were in training |
| 33% exact duplicate rows | the same rows appeared in train and test |
| "Sport" column = which CSV the row came from (football ≈ 94% injured, runners ≈ 1.4%) | a model given **only** the sport column scores almost as well as the full model |
| km, exertion and recovery merged into "duration", "fatigue" and "risk score" columns | features with different meanings were mixed together |
| Accuracy on a 50/50 resampled test set | not meaningful when 96.5% of real rows are "not injured" |

Under a correct protocol the same v1 model drops from F1 0.96 to 0.58. The original notebook is kept for reference.

---

## ⚠️ Known issues

* **Direction of "perceived recovery" is unconfirmed.** The research data stores it on 0–1, but the documentation available here does not say whether higher means *more* or *less* recovered. Within athletes it rises after hard days, which hints that higher = less recovered. The app currently asks "How recovered do you feel?" (higher = better). Check the methods of Lövdal et al. (2021), doi:10.1123/ijspp.2020-0518, before relying on recovery-driven predictions. Details are in `src/uniathleteiq/serving.py`.
* **Unlogged days count as rest days**, so athletes need to log every day for the estimate to be meaningful.
* **No login yet:** the app is single-team and intended for local use until authentication is added.
* **Runners only:** the deployed model is trained on competitive runners.

---

## 📂 Project Structure

```
UniAthleteIQ/
├── notebooks/
│   ├── injury_prediction_pipeline.ipynb   # v1 (original Colab pipeline, kept for reference)
│   ├── 02_leakage_free_pipeline.ipynb     # audit, models, SHAP, calibration, cross-sport test
│   └── 03_statistical_validation.ipynb    # repeated CV, bootstrap CIs, paired tests, baselines
├── src/uniathleteiq/            # research package (shared with the API)
│   ├── data.py          # loaders + feature engineering (workload, personal baselines)
│   ├── serving.py       # daily logs -> model features (same code as training)
│   ├── models.py        # model definitions
│   ├── evaluation.py    # grouped CV, held-out split, metrics, leakage audit
│   ├── stats.py         # athlete bootstrap, paired tests, Holm correction
│   ├── baselines.py     # ACWR / km / exertion reference scores
│   ├── explain.py       # per-prediction explanations + feature labels
│   └── plots.py         # chart styling
├── scripts/
│   ├── run_validation.py          # heavy experiments for notebook 03 (~25 min, cached)
│   ├── train_production_model.py  # model served by the API (all 74 athletes)
│   └── seed_demo.py               # demo athletes for the web app
├── api/                 # FastAPI backend (SQLite locally, Postgres-ready)
├── frontend/            # Next.js web app
├── app/streamlit_app.py # research dashboard for held-out athletes
├── tests/               # leakage guards, stats, serving consistency, API
├── datasets/            # put the Kaggle CSVs here (not committed)
└── outputs/             # charts/, results/, models/, cache/
```

---

## ▶️ How to Run

### 1. Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put the Kaggle CSVs in `datasets/`:

* `day_approach_maskedID_timeseries.csv`
* `week_approach_maskedID_timeseries.csv`
* `dataset.csv`
* `sports_injury_detection_dataset.csv`
* (optional, v1 only) `NBA Player Injury Stats(1951 - 2023).csv`, `basketball_injury_biomechanical_dataset.csv`

### 2. Research

```powershell
jupyter notebook notebooks/02_leakage_free_pipeline.ipynb   # about 5 min
python scripts/run_validation.py                            # about 25 min, then:
jupyter notebook notebooks/03_statistical_validation.ipynb  # about 3 min
pytest                                                      # all tests
```

### 3. Web app

```powershell
python scripts/train_production_model.py   # once: creates the served model
python scripts/seed_demo.py                # optional: 6 demo athletes from the research data
uvicorn api.main:app --reload              # API on http://localhost:8000 (docs at /docs)
```

In a second terminal:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev                                # app on http://localhost:3000
```

---

## 🔮 Roadmap

* Confirm the perceived-recovery scale direction (see Known issues)
* Authentication and multi-team support (Supabase), Docker, cloud deployment
* A second real dataset (e.g. SoccerMon, PMData) for a stronger cross-sport test
* Sequence models (LSTM or temporal CNN) and time-to-injury (survival) modelling
* Paper draft following the TRIPOD+AI reporting checklist
* Prospective pilot with a university team

---

## 👨‍💻 Author

**RAJ ROY**
MCA — Department of Computer Applications

---

## ⭐ Acknowledgements

* Kaggle dataset contributors (competitive runners' training logs, EPL player injuries, multi-sport wearable data)
* scikit-learn, XGBoost, imbalanced-learn, SHAP, FastAPI, SQLModel, Next.js, Recharts, Streamlit
* Research on training load and injury risk (ACWR, training monotony and strain)
