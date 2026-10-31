# 🧠 UniAthleteIQ — AI-Powered Athlete Injury Risk Prediction

## 🚀 Overview

UniAthleteIQ is a machine learning framework that predicts athlete injury risk from training-load, wellness and physiological data, and explains each prediction to coaches.

It works with public datasets from several sports (competitive running, football, and a multi-sport wearable dataset), and it explicitly tests whether injury-risk patterns **transfer across sports**.

**Version 2** rebuilds the evaluation so that every reported number comes from athletes the model has never seen. See [What changed in v2](#-what-changed-in-v2) for why the v1 numbers (96% accuracy) were not reliable.

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
* 🧠 **Explainable AI:** SHAP explanations for every prediction
* 🔒 **Leakage-free evaluation:** athlete-grouped cross-validation plus a held-out set of athletes

---

## 📊 Datasets (Kaggle, public)

| Dataset | Unit | Size | Used for |
|---|---|---|---|
| Competitive runners: daily + weekly training logs | athlete-day with 7-day / 3-week history | 42,680 days, 74 athletes, 575 injuries | **Main risk model** |
| EPL football players | player-season | 1,301 seasons, 604 players | Severe-injury season (more than 28 days out) |
| Multi-sport wearable sessions | session | 1,000 sessions | Secondary; appears synthetic (see notebook §5) |
| Basketball biomechanics, NBA injury list | injured players only | — | Not usable for risk models (no uninjured comparison group) |

---

## ⚙️ Methodology (v2)

1. **Feature engineering (runners):** 7-day load, hard (Z5, interval, sprint) km, rest days, **ACWR** (acute:chronic workload ratio), **monotony** and **strain** (Foster), and perceived exertion, recovery and training-success means and trends.
2. **Personal baselines:** each feature as a z-score against the athlete's *own past days only*. There is no look-ahead, and a unit test enforces this.
3. **Imbalance handling:** class weights. SMOTE is only ever applied inside training folds; it gave no benefit.
4. **Models:** Logistic Regression ⭐ (final), Random Forest, XGBoost.
5. **Evaluation:** 5-fold athlete-grouped CV for model selection, then a single test on 20% of athletes held out from the start. The main metrics are ROC-AUC and PR-AUC on the real class balance (1.35% of runner-days are injuries).
6. **Explainability and calibration:** exact SHAP values for the linear model, Platt-calibrated probabilities, and an alert threshold chosen on training data only.

---

## 🏆 Results (honest, leakage-free)

| Experiment | Evaluated on | ROC-AUC | PR-AUC (random guess) |
|---|---|---|---|
| Runners: Logistic Regression ⭐ | 15 held-out athletes | **0.625** | 0.019 (0.012) |
| Runners: Logistic Regression | athlete-grouped 5-fold CV | 0.654 | 0.026 (0.014) |
| Runners: same athletes, later in time | last 20% of each timeline | 0.620 | 0.031 (0.019) |
| Football: severe-injury season | held-out players | 0.652 | 0.799 (0.708) |
| Multi-sport wearable | 5-fold CV | 0.596 | 0.697 (0.617) |

**What this means in practice:** flagging the riskiest 20% of runner-days catches about **34% of injuries**, compared with 20% for random flagging. The model is useful for prioritising coach check-ins, not for diagnosing individual athletes. Modest AUCs like these are typical of injury prediction from training logs.

### Key findings

* **Personal baselines beat model complexity.** Comparing each athlete with their own normal improved every model (runners CV AUC 0.60–0.63 → 0.65–0.66). Logistic Regression, Random Forest and XGBoost were then tied.
* **Top risk signals:** training **strain above the athlete's own normal**, more hard km, high peak exertion, and a jump in recent km compared with the athlete's usual level.
* **Cross-sport:** with dataset-normalised load and effort, a model trained on one sport ranks injuries in the other about as well as a model trained on that sport itself (ROC-AUC ≈ 0.59–0.61). The signal is weak, and the multi-sport dataset appears synthetic.
* **Data-quality findings:** the runner injury rate depends on how long an athlete has been recorded (a collection artifact the model is prevented from exploiting), and the multi-sport data has the flat distributions typical of synthetic data.

Charts are in [outputs/charts/v2/](outputs/charts/v2/) and result tables in [outputs/results/](outputs/results/).

---

## 🔍 What changed in v2

The v1 notebook reported 96% accuracy (Random Forest). An audit ([notebook §1](notebooks/02_leakage_free_pipeline.ipynb)) showed that this came from data leakage:

| v1 problem | Effect |
|---|---|
| SMOTE applied **before** the train/test split | synthetic near-copies of test rows were in training |
| 33% exact duplicate rows | the same rows appeared in train and test |
| "Sport" column = which CSV the row came from (football ≈ 94% injured, runners ≈ 1.4%) | a model given **only** the sport column scores almost as well as the full model |
| km, exertion and recovery merged into "duration", "fatigue" and "risk score" columns | features with different meanings were mixed together |
| Accuracy on a 50/50 resampled test set | not meaningful when 96.5% of real rows are "not injured" |

Under a correct protocol the same v1 model drops from F1 0.96 to 0.58. "Sport type is the #1 predictor" was an artifact of how the datasets were merged. The original notebook is kept for reference.

---

## 📂 Project Structure

```
UniAthleteIQ/
├── notebooks/
│   ├── injury_prediction_pipeline.ipynb   # v1 (original Colab pipeline, kept for reference)
│   └── 02_leakage_free_pipeline.ipynb     # v2: audit, models, SHAP, cross-sport test
├── src/uniathleteiq/                      # reusable package
│   ├── data.py          # loaders + feature engineering (workload, personal baselines)
│   ├── models.py        # model definitions
│   ├── evaluation.py    # grouped CV, held-out split, metrics, leakage audit
│   ├── explain.py       # per-prediction explanations
│   └── plots.py         # chart styling
├── app/streamlit_app.py # runner risk dashboard
├── tests/               # leakage guards (pytest)
├── datasets/            # put the Kaggle CSVs here (not committed)
└── outputs/             # charts/, results/, models/
```

---

## ▶️ How to Run

### Locally (Windows)

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

Then:

```powershell
jupyter notebook notebooks/02_leakage_free_pipeline.ipynb   # run all cells (about 5 min)
streamlit run app/streamlit_app.py                          # risk dashboard (after the notebook)
pytest                                                      # leakage tests
```

### Google Colab

Upload the project folder (with `datasets/`) to Drive, `%cd` into it, run `!pip install -r requirements.txt`, then run the notebook.

---

## 🔮 Future Scope

* Sequence models (LSTM or temporal CNN) over the full daily history
* Time-to-injury (survival) modelling
* Datasets that record the same load measures (session RPE × duration, GPS) across several sports, for a stronger cross-sport test
* Prospective validation with a real team; IoT wearable integration
* More sports (Cricket, Badminton)

---

## 👨‍💻 Author

**RAJ ROY**
MCA — Department of Computer Applications

---

## ⭐ Acknowledgements

* Kaggle dataset contributors (competitive runners' training logs, EPL player injuries, multi-sport wearable data)
* scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit
* Research on training load and injury risk (ACWR, training monotony and strain)
