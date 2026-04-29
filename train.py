# ============================================================
# HEART DISEASE PREDICTION - END TO END ML PIPELINE
# Dataset: Cleveland Heart Disease (UCI / Kaggle)
# Author: SE-MECH-A | SPPU 2024 Pattern
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# Save as heart.csv in the same folder
df = pd.read_csv("heart.csv")

print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names: {list(df.columns)}")

# ============================================================
# STEP 2: DATA UNDERSTANDING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Data Understanding")
print("=" * 60)

print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
print(f"  0 = No Heart Disease | 1 = Heart Disease")

# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis (EDA)")
print("=" * 60)

# --- Plot 1: Target Distribution ---
plt.figure(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
df['target'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.xlabel('0 = No Disease | 1 = Disease')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/01_target_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/01_target_distribution.png")

# --- Plot 2: Age Distribution by Target ---
plt.figure(figsize=(8, 5))
for target, color, label in zip([0, 1], ['#2ecc71', '#e74c3c'], ['No Disease', 'Heart Disease']):
    plt.hist(df[df['target'] == target]['age'], bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
plt.title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/02_age_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/02_age_distribution.png")

# --- Plot 3: Correlation Heatmap ---
plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn', mask=mask,
            linewidths=0.5, annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: outputs/03_correlation_heatmap.png")

# --- Plot 4: Gender vs Heart Disease ---
plt.figure(figsize=(6, 4))
pd.crosstab(df['sex'], df['target']).plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                                           edgecolor='black', rot=0)
plt.title('Gender vs Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Sex (0=Female, 1=Male)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Heart Disease'])
plt.tight_layout()
plt.savefig('outputs/04_gender_vs_disease.png', dpi=150)
plt.close()
print("Saved: outputs/04_gender_vs_disease.png")

# --- Plot 5: Chest Pain Type vs Target ---
plt.figure(figsize=(7, 4))
pd.crosstab(df['cp'], df['target']).plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                                          edgecolor='black', rot=0)
plt.title('Chest Pain Type vs Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Heart Disease'])
plt.tight_layout()
plt.savefig('outputs/05_chestpain_vs_disease.png', dpi=150)
plt.close()
print("Saved: outputs/05_chestpain_vs_disease.png")

# --- Plot 6: Boxplots for numerical features ---
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 5))
for i, col in enumerate(num_cols):
    df.boxplot(column=col, by='target', ax=axes[i], patch_artist=True)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('Target')
plt.suptitle('Boxplots of Numerical Features vs Target', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/06_boxplots.png', dpi=150)
plt.close()
print("Saved: outputs/06_boxplots.png")

# ============================================================
# STEP 4: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Data Preprocessing & Feature Engineering")
print("=" * 60)

# Drop duplicates if any
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_shape[0] - df.shape[0]}")

# Features and Target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling done (StandardScaler)")

# ============================================================
# STEP 5: MODEL BUILDING & TRAINING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Model Building & Training")
print("=" * 60)

models = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                    SVC(probability=True, random_state=42),
    "KNN":                    KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": roc, "CV Accuracy": cv}
    print(f"\n{name}:")
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ROC-AUC Score : {roc:.4f}")
    print(f"  CV Accuracy   : {cv:.4f}")

# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Model Evaluation & Comparison")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
print(f"\nModel Comparison:\n{results_df.round(4)}")

# Best Model
best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} with accuracy {results_df.loc[best_model_name, 'Accuracy']*100:.2f}%")

# Classification Report for Best Model
y_pred_best = best_model.predict(X_test_scaled)
print(f"\nClassification Report ({best_model_name}):\n")
print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Heart Disease']))

# --- Plot 7: Model Comparison Bar Chart ---
plt.figure(figsize=(10, 5))
results_df[['Accuracy', 'ROC-AUC', 'CV Accuracy']].plot(
    kind='bar', figsize=(10, 5), color=['#3498db', '#e74c3c', '#2ecc71'],
    edgecolor='black', rot=30
)
plt.title('Model Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.ylim(0.7, 1.0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/07_model_comparison.png', dpi=150)
plt.close()
print("Saved: outputs/07_model_comparison.png")

# --- Plot 8: Confusion Matrix of Best Model ---
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Heart Disease'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/08_confusion_matrix.png")

# --- Plot 9: ROC Curves for all models ---
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/09_roc_curves.png', dpi=150)
plt.close()
print("Saved: outputs/09_roc_curves.png")

# --- Plot 10: Feature Importance (Random Forest) ---
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/10_feature_importance.png', dpi=150)
plt.close()
print("Saved: outputs/10_feature_importance.png")

# ============================================================
# STEP 7: HYPERPARAMETER TUNING (Best Model)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Hyperparameter Tuning (Random Forest)")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score:   {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test_scaled)
final_acc = accuracy_score(y_test, y_pred_final)
print(f"Final Model Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

# ============================================================
# STEP 8: SAVE MODEL & SCALER
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Model & Scaler")
print("=" * 60)

with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Saved: model/heart_model.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/feature_names.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE! All outputs saved.")
print("   Run: streamlit run app.py")
print("=" * 60)
