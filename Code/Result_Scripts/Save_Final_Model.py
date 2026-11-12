import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_folder = os.path.join(project_folder, "Processed_Data")
results_folder = os.path.join(project_folder, "Results")
os.makedirs(results_folder, exist_ok=True)
X_train = pd.read_csv(
    os.path.join(data_folder, "X_train_raw.csv")
)
y_train = pd.read_csv(
    os.path.join(data_folder, "Y_train_original.csv")
).values.ravel()
final_model = Pipeline([
    ("scaler", StandardScaler()),

    ("oversampler", RandomOverSampler(
        sampling_strategy=0.36,
        random_state=42
    )),

    ("classifier", GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])
print("Training final Gradient Boosting model...")
final_model.fit(X_train, y_train)
model_artifact = {
    "model": final_model,
    "threshold": 0.375,
    "model_name": "Tuned Gradient Boosting",
    "features": list(X_train.columns)
}
model_path = os.path.join(
    results_folder,
    "final_diabetes_model.pkl"
)
joblib.dump(model_artifact, model_path)
print("\nFinal model saved successfully!")
print("Model path:", model_path)
print("Selected threshold:", model_artifact["threshold"])
print("Number of features:", len(model_artifact["features"]))