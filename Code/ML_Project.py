import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
project_folder = Path(__file__).resolve().parent.parent
dataset_path = (
    project_folder
    / "Dataset"
    / "diabetes_012_health_indicators_BRFSS2015.csv.zip"
)
df = pd.read_csv(dataset_path)
print("Original dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe().T)

print("\nOriginal target distribution:")
print(df["Diabetes_012"].value_counts().sort_index())

df = df[df["Diabetes_012"].isin([0, 2])].copy()

df["Diabetes"] = df["Diabetes_012"].map({
    0: 0,
    2: 1
})

df = df.drop(columns=["Diabetes_012"])

print("\nPrepared dataset shape:", df.shape)

print("\nTarget class counts:")
print(df["Diabetes"].value_counts().sort_index())

print("\nTarget class percentages:")
print(
    df["Diabetes"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nMissing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nPrepared dataset preview:")
print(df.head())

print("\nProgram completed successfully!")

target_counts = df["Diabetes"].value_counts().sort_index()

print("\nTarget variable distribution:")
print(target_counts)

target_counts.plot(kind="bar")
plt.title("Distribution of Diabetes Classes")
plt.xlabel("Diabetes Status")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"], rotation=0)
plt.tight_layout()
plt.show()