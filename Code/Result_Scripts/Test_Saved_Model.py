import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_folder = os.path.join(project_folder, "Processed_Data")
results_folder = os.path.join(project_folder, "Results")
model_path = os.path.join(
    results_folder,
    "final_diabetes_model.pkl"
)
artifact = joblib.load(model_path)
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
X_test = pd.read_csv(
    os.path.join(data_folder, "X_test_raw.csv")
)
y_test = pd.read_csv(
    os.path.join(data_folder, "Y_test.csv")
).values.ravel()
X_test = X_test[features]
probabilities = model.predict_proba(X_test)[:, 1]
y_pred = (probabilities >= threshold).astype(int)
print("Saved model loaded successfully!")
print("Model:", artifact["model_name"])
print("Threshold:", threshold)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))