# ❤️ Heart Disease Prediction System
## End-to-End AI/ML Project | SE Mechanical Engineering | SPPU 2024 Pattern

---

## 📁 Project Structure

```
heart_disease_prediction/
│
├── heart.csv              ← Dataset (download from Kaggle)
├── train.py               ← Full ML pipeline (run this first)
├── app.py                 ← Streamlit web app
├── requirements.txt       ← Python dependencies
│
├── model/                 ← Auto-created after training
│   ├── heart_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── outputs/               ← Auto-created EDA & evaluation plots
    ├── 01_target_distribution.png
    ├── 02_age_distribution.png
    ├── 03_correlation_heatmap.png
    ├── 04_gender_vs_disease.png
    ├── 05_chestpain_vs_disease.png
    ├── 06_boxplots.png
    ├── 07_model_comparison.png
    ├── 08_confusion_matrix.png
    ├── 09_roc_curves.png
    └── 10_feature_importance.png
```

---

## 📦 Dataset

1. Go to: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Download `heart.csv`
3. Place it in the root project folder

**Features (13 input + 1 target):**

| Feature   | Description                                    |
|-----------|------------------------------------------------|
| age       | Age in years                                   |
| sex       | 1=Male, 0=Female                               |
| cp        | Chest pain type (0–3)                          |
| trestbps  | Resting blood pressure (mmHg)                  |
| chol      | Serum cholesterol (mg/dl)                      |
| fbs       | Fasting blood sugar > 120 mg/dl (1=Yes, 0=No) |
| restecg   | Resting ECG results (0–2)                      |
| thalach   | Maximum heart rate achieved                    |
| exang     | Exercise induced angina (1=Yes, 0=No)          |
| oldpeak   | ST depression induced by exercise              |
| slope     | Slope of peak exercise ST segment (0–2)        |
| ca        | Number of major vessels (0–3)                  |
| thal      | Thalassemia (1=Normal, 2=Fixed, 3=Reversible)  |
| target    | 0=No Disease, 1=Heart Disease                  |

---

## ⚙️ Setup & Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training pipeline
python train.py

# 4. Launch web app
streamlit run app.py
```

---

## 🤖 Models Used

| Model                | Tuned |
|---------------------|-------|
| Logistic Regression  | No    |
| Random Forest        | ✅ Yes (GridSearchCV) |
| Support Vector Machine | No  |
| K-Nearest Neighbors  | No   |
| Gradient Boosting    | No   |

---

## 📊 Pipeline Steps

1. **Data Loading** — Load Cleveland dataset
2. **Data Understanding** — Shape, dtypes, missing values
3. **EDA** — 6 visualizations (distributions, heatmap, boxplots)
4. **Preprocessing** — Duplicate removal, train-test split (80:20), StandardScaler
5. **Model Training** — 5 models trained & compared
6. **Evaluation** — Accuracy, ROC-AUC, Cross-validation, Confusion Matrix
7. **Hyperparameter Tuning** — GridSearchCV on Random Forest
8. **Deployment** — Streamlit app with real-time prediction

---

## 📈 Expected Results

- **Best Model Accuracy:** ~88–92%
- **ROC-AUC Score:** ~0.92–0.96

---

## 📄 For SPPU Project Report

Chapters to include:
1. Abstract
2. Introduction & Problem Statement
3. Literature Review
4. Dataset Description
5. Methodology / System Architecture
6. EDA & Results
7. Model Comparison
8. Conclusion & Future Scope
9. References

---

*⚠️ Disclaimer: This project is for educational purposes only. Not a substitute for medical diagnosis.*
