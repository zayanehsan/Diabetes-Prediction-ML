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
target_counts = df["Diabetes"].value_counts().sort_index()
print("\nTarget variable distribution:")
print(target_counts)
target_counts.plot(kind="bar")
plt.title("Distribution of Diabetes Classes")
plt.xlabel("Diabetes Status")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"], rotation=0)
plt.tight_layout()
plt.savefig("Graphs/diabetes_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nBMI statistical summary:")
print(df["BMI"].describe())
plt.figure(figsize=(8, 5))
plt.hist(df["BMI"], bins=30, edgecolor="black")
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("Graphs/bmi_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Diabetes"].map({0: "No Diabetes", 1: "Diabetes"}), y=df["BMI"])
plt.title("BMI Comparison by Diabetes Status")
plt.xlabel("Diabetes Status")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("Graphs/bmi_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Age")
plt.title("Distribution of Age Categories")
plt.xlabel("Age Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("Graphs/age_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
age_diabetes_percent = pd.crosstab(
    df["Age"],
    df["Diabetes"],
    normalize="index"
) * 100
age_diabetes_percent = age_diabetes_percent.rename(
    columns={0: "No Diabetes", 1: "Diabetes"}
)
age_diabetes_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=["steelblue", "darkorange"]
)
plt.title("Diabetes Status by Age Category")
plt.xlabel("Age Category")
plt.ylabel("Percentage of Records")
plt.legend(title="Diabetes Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Graphs/age_vs_diabetes.png", dpi=300, bbox_inches="tight")
plt.show()
health_diabetes_percent = pd.crosstab(
    df["GenHlth"],
    df["Diabetes"],
    normalize="index"
) * 100
health_diabetes_percent = health_diabetes_percent.rename(
    columns={0: "No Diabetes", 1: "Diabetes"}
)
health_diabetes_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5),
    color=["steelblue", "darkorange"]
)
plt.title("Diabetes Status by General Health")
plt.xlabel("General Health Category")
plt.ylabel("Percentage of Records")
plt.legend(title="Diabetes Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Graphs/general_health_vs_diabetes.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="PhysActivity")
plt.title("Distribution of Physical Activity")
plt.xlabel("Physical Activity (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No Physical Activity", "Physical Activity"])
plt.tight_layout()
plt.savefig("Graphs/physical_activity_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
activity_diabetes_percent = pd.crosstab(
    df["PhysActivity"],
    df["Diabetes"],
    normalize="index"
) * 100
activity_diabetes_percent = activity_diabetes_percent.rename(
    index={0: "No Physical Activity", 1: "Physical Activity"},
    columns={0: "No Diabetes", 1: "Diabetes"}
)
activity_diabetes_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=["steelblue", "darkorange"]
)
plt.title("Diabetes Status by Physical Activity")
plt.xlabel("Physical Activity")
plt.ylabel("Percentage of Records")
plt.legend(title="Diabetes Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Graphs/physical_activity_vs_diabetes.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="HighBP")
plt.title("Distribution of High Blood Pressure")
plt.xlabel("High Blood Pressure")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No High BP", "High BP"])
plt.tight_layout()
plt.savefig("Graphs/high_bp_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
bp_diabetes_percent = pd.crosstab(
    df["HighBP"],
    df["Diabetes"],
    normalize="index"
) * 100
bp_diabetes_percent = bp_diabetes_percent.rename(
    index={0: "No High BP", 1: "High BP"},
    columns={0: "No Diabetes", 1: "Diabetes"}
)
bp_diabetes_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=["steelblue", "darkorange"]
)
plt.title("Diabetes Status by High Blood Pressure")
plt.xlabel("High Blood Pressure")
plt.ylabel("Percentage of Records")
plt.legend(title="Diabetes Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Graphs/high_bp_vs_diabetes.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="HighChol")
plt.title("Distribution of High Cholesterol")
plt.xlabel("High Cholesterol")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No High Cholesterol", "High Cholesterol"])
plt.tight_layout()
plt.savefig("Graphs/high_cholesterol_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
chol_diabetes_percent = pd.crosstab(
    df["HighChol"],
    df["Diabetes"],
    normalize="index"
) * 100
chol_diabetes_percent = chol_diabetes_percent.rename(
    index={0: "No High Cholesterol", 1: "High Cholesterol"},
    columns={0: "No Diabetes", 1: "Diabetes"}
)
chol_diabetes_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=["steelblue", "darkorange"]
)
plt.title("Diabetes Status by High Cholesterol")
plt.xlabel("High Cholesterol")
plt.ylabel("Percentage of Records")
plt.legend(title="Diabetes Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Graphs/high_cholesterol_vs_diabetes.png", dpi=300, bbox_inches="tight")
plt.show()
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("Graphs/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
Q1 = df["BMI"].quantile(0.25)
Q3 = df["BMI"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
bmi_outliers = df[
    (df["BMI"] < lower_bound) |
    (df["BMI"] > upper_bound)
]
print("\nBMI Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower boundary:", lower_bound)
print("Upper boundary:", upper_bound)
print("Number of potential BMI outliers:", len(bmi_outliers))
print("Percentage of potential BMI outliers:", round(len(bmi_outliers) / len(df) * 100, 2), "%")
exact_duplicates = df.duplicated().sum()
feature_duplicates = df.drop(
    columns=["Diabetes"]
).duplicated().sum()
print("\nDuplicate Analysis")
print("Exact duplicate rows:", exact_duplicates)
print("Duplicate input-feature rows:", feature_duplicates)
Q1 = df["MentHlth"].quantile(0.25)
Q3 = df["MentHlth"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
menthlth_outliers = df[
    (df["MentHlth"] < lower_bound) |
    (df["MentHlth"] > upper_bound)
]
print("\nMental Health Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower boundary:", lower_bound)
print("Upper boundary:", upper_bound)
print("Number of potential outliers:", len(menthlth_outliers))
print("Percentage of potential outliers:", round(len(menthlth_outliers) / len(df) * 100, 2), "%")
Q1 = df["PhysHlth"].quantile(0.25)
Q3 = df["PhysHlth"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
physhlth_outliers = df[
    (df["PhysHlth"] < lower_bound) |
    (df["PhysHlth"] > upper_bound)
]
print("\nPhysical Health Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower boundary:", lower_bound)
print("Upper boundary:", upper_bound)
print("Number of potential outliers:", len(physhlth_outliers))
print("Percentage of potential outliers:", round(len(physhlth_outliers) / len(df) * 100, 2), "%")
print("\nValid Value and Range Analysis")
print("\nBMI range:")
print("Minimum:", df["BMI"].min())
print("Maximum:", df["BMI"].max())
print("Invalid BMI values:", ((df["BMI"] <= 0) | (df["BMI"] > 100)).sum())
print("\nAge category range:")
print("Minimum:", df["Age"].min())
print("Maximum:", df["Age"].max())
print("Invalid Age categories:", ((df["Age"] < 1) | (df["Age"] > 13)).sum())
print("\nPhysical health range:")
print("Minimum:", df["PhysHlth"].min())
print("Maximum:", df["PhysHlth"].max())
print("Invalid values:", ((df["PhysHlth"] < 0) | (df["PhysHlth"] > 30)).sum())
print("\nMental health range:")
print("Minimum:", df["MentHlth"].min())
print("Maximum:", df["MentHlth"].max())
print("Invalid values:", ((df["MentHlth"] < 0) | (df["MentHlth"] > 30)).sum())
print("\nTarget values:")
print(df["Diabetes"].value_counts().sort_index())
print("Invalid target values:", (~df["Diabetes"].isin([0, 1])).sum())
feature_columns = df.drop(columns=["Diabetes"]).columns
feature_groups = df.groupby(list(feature_columns))["Diabetes"].nunique()
conflicting_groups = feature_groups[feature_groups > 1]
print("\nConflicting Target Label Analysis")
print("Feature combinations with conflicting labels:", len(conflicting_groups))
print("Rows belonging to conflicting groups:", feature_groups[feature_groups > 1].sum())
features_to_compare = ["Smoker", "HeartDiseaseorAttack", "DiffWalk"]
for feature in features_to_compare:
    comparison = pd.crosstab(
        df[feature],
        df["Diabetes"],
        normalize="index"
    ) * 100
    comparison.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=["skyblue", "salmon"]
    )
    plt.title(f"{feature} vs Diabetes")
    plt.xlabel(feature)
    plt.ylabel("Percentage of respondents")
    plt.legend(["No Diabetes", "Diabetes"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"Graphs/{feature.lower()}_vs_diabetes.png", dpi=300, bbox_inches="tight")
    plt.show()
features_to_compare = ["Smoker", "HeartDiseaseorAttack", "DiffWalk"]
for feature in features_to_compare:
    print(f"\n{feature} vs Diabetes")
    comparison = pd.crosstab(
        df[feature],
        df["Diabetes"],
        normalize="index"
    ) * 100
    print(comparison.round(2))