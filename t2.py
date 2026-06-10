import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: File Path (update if your file is elsewhere)
file_path = r"e:/Internships/saiket data analyst intern/task 2/t2.py"

if os.path.exists(file_path):
    print(" File found!")
else:
    print("File not found:", file_path)
    exit()  # stop the program if file is missing

#  Step 3: Read CSV file
df = pd.read_csv(file_path)

# Step 4: Data cleaning
# Convert 'TotalCharges' to numeric and fill missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ----------------------------------------------------------
# 1️ Overall Churn Rate
# ----------------------------------------------------------
churn_rate = df["Churn"].value_counts(normalize=True) * 100
plt.figure(figsize=(5, 5))
churn_rate.plot(kind="pie", autopct="%.1f%%", labels=["No Churn", "Churn"], colors=["skyblue", "salmon"])
plt.title("Overall Customer Churn Rate")
plt.ylabel("")
plt.show()

# ----------------------------------------------------------
# 2️ Customer Distribution by Gender, Partner, Dependents
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df["gender"].value_counts().plot(kind="bar", ax=axes[0], color="skyblue", title="Customer Distribution by Gender")
df["Partner"].value_counts().plot(kind="bar", ax=axes[1], color="lightgreen", title="Customer Distribution by Partner Status")
df["Dependents"].value_counts().plot(kind="bar", ax=axes[2], color="orange", title="Customer Distribution by Dependents")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3️ Tenure Distribution and Relation with Churn
# ----------------------------------------------------------
plt.figure(figsize=(8, 5))
df[df["Churn"] == "No"]["tenure"].plot(kind="hist", bins=30, alpha=0.7, label="No Churn", color="green")
df[df["Churn"] == "Yes"]["tenure"].plot(kind="hist", bins=30, alpha=0.7, label="Churn", color="red")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()
plt.show()

# ----------------------------------------------------------
# 4️ Churn Rate by Contract Type
# ----------------------------------------------------------
contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack() * 100
contract_churn.plot(kind="bar", figsize=(8, 5), title="Churn Rate by Contract Type", stacked=True)
plt.ylabel("Percentage")
plt.show()

# ----------------------------------------------------------
# 5️ Churn Rate by Payment Method
# ----------------------------------------------------------
payment_churn = df.groupby("PaymentMethod")["Churn"].value_counts(normalize=True).unstack() * 100
payment_churn.plot(kind="bar", figsize=(10, 5), title="Churn Rate by Payment Method", stacked=True)
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# ----------------------------------------------------------
# Summary Output
# ----------------------------------------------------------
print("\n--- Overall Churn Rate ---")
print(churn_rate)
print("\nEDA Completed Successfully!")
