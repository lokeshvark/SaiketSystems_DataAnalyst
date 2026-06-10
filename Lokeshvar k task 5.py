
import os
import glob
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def load_csv_automatically():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))

    if not csv_files:
        print("❌ No CSV file found in the script folder.")
        print("👉 Please put your Telco churn CSV in the SAME folder as this .py file and run again.")
        sys.exit(0)
        
    preferred = None
    for f in csv_files:
        name = os.path.basename(f).lower()
        if "churn" in name or "telco" in name:
            preferred = f
            break

    csv_path = preferred if preferred is not None else csv_files[0]

    print(f"✅ Using CSV file: {os.path.basename(csv_path)}")
    return pd.read_csv(csv_path)

df = load_csv_automatically()
print("✅ Dataset loaded. Shape:", df.shape)
print(df.head())

required_cols = ["Churn", "TotalCharges", "MonthlyCharges"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Required column '{col}' not found in the dataset.")
        print("👉 Please make sure you are using the Telco Customer Churn dataset.")
        sys.exit(0)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

if df["ChurnFlag"].isna().any():
    print("❌ Unexpected values in 'Churn' column. Expected only 'Yes' or 'No'.")
    sys.exit(0)

drop_cols = [c for c in ["customerID", "Churn", "ChurnFlag"] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["ChurnFlag"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\n📊 Numeric columns:", num_cols)
print("📊 Categorical columns:", cat_cols)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline([
    ("prep", preprocess),
    ("clf", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

print("\n✅ Model trained successfully.")
print("🔹 Training Accuracy:", pipeline.score(X_train, y_train))
print("🔹 Testing Accuracy :", pipeline.score(X_test, y_test))

df["churn_probability"] = pipeline.predict_proba(X)[:, 1]

print("\n🎯 Sample churn probabilities:")
sample_cols = [c for c in ["customerID", "Churn", "churn_probability"] if c in df.columns]
print(df[sample_cols].head())

try:
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    ohe_features = ohe.get_feature_names_out(cat_cols)
    all_features = np.concatenate([num_cols, ohe_features])

    coefs = pipeline.named_steps["clf"].coef_[0]

    imp = pd.DataFrame({
        "feature": all_features,
        "coefficient": coefs
    })
    imp["abs_coeff"] = imp["coefficient"].abs()
    imp = imp.sort_values(by="abs_coeff", ascending=False)

    print("\n🏆 Top 10 most important features for churn:")
    print(imp.head(10)[["feature", "coefficient"]])

    imp.to_csv("churn_feature_importance.csv", index=False)
    print("💾 Saved: churn_feature_importance.csv")

except Exception as e:
    print("\n⚠ Could not compute detailed feature importance.")
    print("Reason:", str(e))

monthly_revenue = df["MonthlyCharges"].mean()
churn_rate = df["ChurnFlag"].mean()

print("\n💰 Average Monthly Revenue (ARPU):", monthly_revenue)
print("📉 Monthly Churn Rate:", churn_rate)

if churn_rate > 0:
    ltv = monthly_revenue / churn_rate
else:
    ltv = None

print("📌 Estimated Overall Lifetime Value (LTV):", ltv)

if "Contract" in df.columns:
    segment_stats = (
        df.groupby("Contract")
        .agg(
            customers=("ChurnFlag", "count"),
            churn_rate=("ChurnFlag", "mean"),
            arpu=("MonthlyCharges", "mean")
        )
        .reset_index()
    )

    segment_stats["ltv_estimated"] = np.where(
        segment_stats["churn_rate"] > 0,
        segment_stats["arpu"] / segment_stats["churn_rate"],
        np.nan
    )

    print("\n📊 Segment-wise LTV by Contract:")
    print(segment_stats)

    ltv_map = dict(zip(segment_stats["Contract"], segment_stats["ltv_estimated"]))
    df["segment_LTV"] = df["Contract"].map(ltv_map)
else:
    print("\n⚠ Column 'Contract' not found. Skipping segment-wise LTV.")
    df["segment_LTV"] = np.nan

value_threshold = df["TotalCharges"].quantile(0.75)
df["is_high_value"] = df["TotalCharges"] >= value_threshold

risk_threshold = 0.6
df["is_high_risk"] = df["churn_probability"] > risk_threshold

high_value_at_risk = df[df["is_high_value"] & df["is_high_risk"]]

print("\n🚨 Number of HIGH-VALUE customers at RISK of churn:", high_value_at_risk.shape[0])

cols_to_show = [c for c in ["customerID", "TotalCharges", "MonthlyCharges",
                            "Contract", "churn_probability"] if c in df.columns]

print(high_value_at_risk[cols_to_show].head(20))

high_value_at_risk.to_csv("high_value_customers_at_risk.csv", index=False)
print("\n💾 Saved: high_value_customers_at_risk.csv")

print("\n✅ Task 5 completed: model, LTV, key factors, and high-value at-risk customers generated.")

