# 🧠 UniAthleteIQ — AI-Powered Athlete Injury Risk Prediction

## 🚀 Overview

UniAthleteIQ is a **cross-sport machine learning framework** designed to predict athlete injury risk using physiological and training data.

Unlike traditional models that focus on a single sport, this system generalizes across multiple sports such as **Football, Basketball, and Athletics**, making it more practical for real-world use.

---

## 🎯 Problem Statement

* Sports injuries are often detected **after they occur**
* Most ML models are **sport-specific**
* Coaches lack **interpretable insights** from predictions

---

## 💡 Proposed Solution

UniAthleteIQ introduces:

* 🔮 **Early injury prediction**
* 🌍 **Cross-sport generalization**
* 🧠 **Explainable AI (SHAP)**
* 📊 **Data harmonization pipeline**

---

## 📊 Dataset

* 4 public datasets (Kaggle)
* Total records: **87,865**
* Sports covered:

  * Football
  * Basketball
  * Athletics
  * Multi-sport collegiate data

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Column standardization
* Label encoding
* Missing value handling

### 2. Data Harmonization

* Merged multiple datasets into a unified schema

### 3. Class Imbalance Handling

* Applied **SMOTE**
* Balanced ratio: 96.5% → 50%

### 4. Feature Engineering

* Heart Rate
* Fatigue Index
* Duration
* Injury Risk Score
* Sport Type

### 5. Model Training

* Logistic Regression
* Random Forest ⭐
* XGBoost

---

## 🏆 Results

| Model               | Accuracy   | F1 Score   | AUC       |
| ------------------- | ---------- | ---------- | --------- |
| Logistic Regression | 80.69%     | 0.7625     | 0.847     |
| Random Forest ⭐     | **96.38%** | **0.9629** | **0.991** |
| XGBoost             | 92.54%     | 0.9202     | 0.968     |

👉 **Random Forest achieved the best performance**

---

## 📈 Key Insight

* **Sport Type is the #1 predictor (30% importance)**
* Confirms that injury risk varies significantly across sports

---

## 🧠 Explainable AI

* SHAP values used to interpret predictions
* Helps coaches understand:

  * Why injury risk is high
  * Which features contributed

---

## 🛠 Tech Stack

* Python
* scikit-learn
* XGBoost
* SMOTE (imblearn)
* SHAP
* Google Colab
* Kaggle datasets

---

## 📂 Project Structure

```
UniAthleteIQ/
│
├── notebooks/       # Model training (Colab)
├── paper/           # Research paper
├── ppt/             # Presentation
├── outputs/         # Charts & results
└── README.md
```

---

## ▶️ How to Run

### Locally (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/injury_prediction_pipeline.ipynb
```

Put the Kaggle CSVs in `datasets/` before running:

* `sports_injury_detection_dataset.csv`
* `day_approach_maskedID_timeseries.csv`
* `week_approach_maskedID_timeseries.csv`
* `dataset.csv`
* (optional, only previewed) `NBA Player Injury Stats(1951 - 2023).csv`, `basketball_injury_biomechanical_dataset.csv`

Charts are written to `outputs/charts/`.

### Google Colab

1. Upload the project folder (with `datasets/`) to Drive and `%cd` into it
2. Run all cells step-by-step

---

## 🔮 Future Scope

* LSTM for time-series prediction
* IoT wearable integration
* Web-based dashboard
* Add more sports (Cricket, Badminton)

---

## 👨‍💻 Author

**RAJ ROY**
MCA — Department of Computer Applications

---

## ⭐ Acknowledgements

* Kaggle datasets
* Open-source ML libraries
* Research papers in sports analytics
