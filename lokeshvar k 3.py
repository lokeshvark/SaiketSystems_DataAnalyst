
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


search_patterns = [
    r"C:\Users\lokes\OneDrive\Desktop\saiket data analyst intern\Telco_Customer_Churn_Dataset*.csv",
    r"C:\Users\lokes\OneDrive\Desktop\saiket data analyst intern\*.csv"
]

dataset_path = None


for pattern in search_patterns:
    matches = glob.glob(pattern)
    if matches:
        dataset_path = matches[0]  
        break


if not dataset_path or not os.path.exists(dataset_path):
    raise FileNotFoundError(
        "❌ Dataset not found. Please ensure the file exists in:\n"
        "C:\\Users\\lokes\\OneDrive\\Desktop\\saiket data analyst intern\\\n"
        "and is named something like 'Telco_Customer_Churn_Dataset.csv'"
    )


df = pd.read_csv(dataset_path)
print(f"\n✅ Dataset Loaded Successfully from: {dataset_path}")
print("Shape of dataset:", df.shape)
print(df.head())


print("\n🔍 Checking Missing Values Before Cleaning:\n", df.isnull().sum())


if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("✅ 'TotalCharges' column cleaned successfully.")
else:
    print("⚠️ 'TotalCharges' column not found, skipping conversion.")


duplicates = df.duplicated().sum()
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"✅ Removed {duplicates} duplicate rows.")
else:
    print("✅ No duplicate rows found.")


critical_cols = ['tenure', 'MonthlyCharges', 'Contract', 'Churn']
df.dropna(subset=[col for col in critical_cols if col in df.columns], inplace=True)

print("\n✅ Missing Values After Cleaning:\n", df.isnull().sum())


df = df[critical_cols]
df['Contract'] = df['Contract'].astype(str).str.lower()
df['Churn'] = df['Churn'].astype(str).str.lower()


bins = [0, 12, 24, 48, 72]
labels = ['0-1 yr', '1-2 yrs', '2-4 yrs', '4-6 yrs']
df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)


segment_churn = (
    df.groupby(['TenureGroup', 'Contract'])['Churn']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

print("\n📊 Churn Rate by Tenure Group and Contract Type:\n")
print(segment_churn)


plt.figure(figsize=(10,6))
segment_churn['yes'].unstack().plot(kind='bar', figsize=(10,6))
plt.title('Churn Rate by Tenure and Contract Type')
plt.xlabel('Tenure Group')
plt.ylabel('Churn Rate')
plt.legend(title='Contract Type')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Overall Customer Churn Distribution')
plt.ylabel('')
plt.show()


avg_charges = df.groupby('Contract')['MonthlyCharges'].mean().sort_values(ascending=False)
print("\n💰 Average Monthly Charges by Contract Type:\n", avg_charges)

plt.figure(figsize=(6,4))
sns.barplot(x=avg_charges.index, y=avg_charges.values)
plt.title('Average Monthly Charges by Contract Type')
plt.ylabel('Average Charges ($)')
plt.xlabel('Contract Type')
plt.tight_layout()
plt.show()

print("\n📈 Insights Summary:")
print("- Month-to-month customers tend to have the highest churn rate.")
print("- Customers with longer-term contracts (1 or 2 years) are less likely to churn.")
print("- New customers (0–1 yr tenure) show significantly higher churn risk.")
print("- Higher monthly charges are associated with flexible (month-to-month) plans.")
print("\n✅ Customer Segmentation and Churn Analysis Completed Successfully!")
