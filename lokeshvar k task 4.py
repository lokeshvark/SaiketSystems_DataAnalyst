import os, glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# FILE PATH
orig_path = r"C:\Users\lokes\OneDrive\Desktop\saiket data analyst intern\Telco_Customer_Churn_Dataset .csv"
new_path  = r"C:\Users\lokes\OneDrive\Desktop\saiket data analyst intern\Telco_Customer_Churn_Dataset.csv"

if os.path.exists(orig_path) and not os.path.exists(new_path):
    os.rename(orig_path, new_path)

dataset_path = new_path
df = pd.read_csv(dataset_path)

# CLEANING
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = df.drop_duplicates()

df["Churn"] = df["Churn"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

numeric_features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] if c in df.columns]
categorical_candidates = ["Contract", "PaymentMethod", "InternetService", "gender", "Partner", "Dependents"]
categorical_features = [c for c in categorical_candidates if c in df.columns]

y = df["Churn"]
X = df[numeric_features + categorical_features].copy()

for c in numeric_features:
    X[c] = X[c].fillna(X[c].median())

for c in categorical_features:
    X[c] = X[c].astype(str).fillna("Unknown")

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# PIPELINE
num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

pre = ColumnTransformer([
    ("num", num_pipe, numeric_features),
    ("cat", cat_pipe, categorical_features)
])

clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

# SAFE GRID SEARCH (no parallel workers)

params = {
    "clf__C": [0.1, 1],
    "clf__penalty": ["l1", "l2"]
}

grid = GridSearchCV(pipe, params, cv=3, scoring="roc_auc", n_jobs=1)   # n_jobs=1 fixes crash
grid.fit(X_train, y_train)

best = grid.best_estimator_


# METRICS
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("Best params:", grid.best_params_)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

# SAVE MODEL
joblib.dump(best, "churn_logreg_model.joblib")
print("Model saved as churn_logreg_model.joblib")


