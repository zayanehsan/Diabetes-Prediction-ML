import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
dataset_path = "Dataset/diabetes_012_health_indicators_BRFSS2015.csv.zip"
df = pd.read_csv(dataset_path)
df = df[df["Diabetes_012"] != 1].copy()
df["Diabetes"] = df["Diabetes_012"].map({0: 0, 2: 1})
df.drop(columns=["Diabetes_012"], inplace=True)
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()
print("Exact duplicate rows before removal:", duplicates_before)
print("Exact duplicate rows after removal:", duplicates_after)
print("Dataset shape after duplicate removal:", df.shape)
print("Binary target prepared successfully")
print("Dataset shape:", df.shape)
class_counts = df["Diabetes"].value_counts().sort_index()
class_percentages = df["Diabetes"].value_counts(normalize=True).sort_index() * 100
print("Class distribution:")
print(class_counts)
print("Class percentages:")
print(class_percentages)
print("Minority class percentage:", class_percentages.min())
print("Majority class percentage:", class_percentages.max())
X = df.drop(columns=["Diabetes"])
y = df["Diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled training features shape:", X_train_scaled.shape)
print("Scaled testing features shape:", X_test_scaled.shape)
ros = RandomOverSampler(
    sampling_strategy={1: 47912},
    random_state=42
)
X_train_resampled, y_train_resampled = ros.fit_resample(
    X_train_scaled,
    y_train
)
print("Training class counts before oversampling:")
print(y_train.value_counts().sort_index())
print("Training class counts after oversampling:")
print(y_train_resampled.value_counts().sort_index())
print("Resampled training features shape:", X_train_resampled.shape)
print("Testing features shape:", X_test_scaled.shape)
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target distribution:")
print(y_train.value_counts(normalize=True).sort_index() * 100)
print("Testing target distribution:")
print(y_test.value_counts(normalize=True).sort_index() * 100)
print("BMI range:", df["BMI"].min(), "to", df["BMI"].max())
print("BMI values <= 0:", (df["BMI"] <= 0).sum())
print("MentHlth range:", df["MentHlth"].min(), "to", df["MentHlth"].max())
print("MentHlth values outside 0-30:", ((df["MentHlth"] < 0) | (df["MentHlth"] > 30)).sum())
print("PhysHlth range:", df["PhysHlth"].min(), "to", df["PhysHlth"].max())
print("PhysHlth values outside 0-30:", ((df["PhysHlth"] < 0) | (df["PhysHlth"] > 30)).sum())
output_folder = "Processed_Data"
os.makedirs(output_folder, exist_ok=True)
X_train_resampled_df = pd.DataFrame(
    X_train_resampled,
    columns=X.columns
)
X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)
X_train_resampled_df.to_csv(
    os.path.join(output_folder, "X_train.csv"),
    index=False
)
y_train_resampled.to_csv(
    os.path.join(output_folder, "Y_train.csv"),
    index=False
)
X_test_scaled_df.to_csv(
    os.path.join(output_folder, "X_test.csv"),
    index=False
)
y_test.to_csv(
    os.path.join(output_folder, "Y_test.csv"),
    index=False
)
print("Processed datasets saved successfully.")
