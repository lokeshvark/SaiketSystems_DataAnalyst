import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\lokes\Downloads\Telco_Customer_Churn_Dataset .csv"
df = pd.read_csv(file_path)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

df["ChurnFlag"] = df["Churn"].map({"Yes":1, "No":0})

sns.set(style="whitegrid")

plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure(figsize=(8,5))
sns.violinplot(x="Churn", y="TotalCharges", data=df, inner="quartile")
plt.title("Total Charges Distribution by Churn")
plt.show()

sns.pairplot(df[["tenure","MonthlyCharges","TotalCharges","Churn"]], hue="Churn", diag_kind="kde")
plt.show()
