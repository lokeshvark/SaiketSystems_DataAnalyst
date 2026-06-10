import pandas as pd


df = pd.read_csv(r"C:\Users\lokes\OneDrive\Desktop\saiket data analyst intern\Telco_Customer_Churn_Dataset .csv")

print("Initial Data Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop("customerID", axis=1, inplace=True)

df_encoded = pd.get_dummies(df, drop_first=True)

print("\nCleaned Data Info:")
print(df_encoded.info())

df_encoded.to_csv(r"C:\Users\lokes\OneDrive\Desktop\Cleaned_Telco_Churn.csv", index=False)
print("\n✅ Cleaned dataset saved as 'Cleaned_Telco_Churn.csv'")
