import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_folder = os.path.join(project_folder, "Processed_Data")
results_folder = os.path.join(project_folder, "Results")
X_train = pd.read_csv(
    os.path.join(data_folder, "X_train_raw.csv")
)
y_train = pd.read_csv(
    os.path.join(data_folder, "Y_train_original.csv")
).values.ravel()
X_test = pd.read_csv(
    os.path.join(data_folder, "X_test_raw.csv")
)
y_test = pd.read_csv(
    os.path.join(data_folder, "Y_test.csv")
).values.ravel()
model = Pipeline([
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
print("Training Tuned Gradient Boosting...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))
print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)
total = len(y_test)
print("\n===== ERROR ANALYSIS =====")
print("Total Test Samples:", total)
print("Correct Predictions:", tn + tp)
print("Incorrect Predictions:", fp + fn)
print("\nFalse Positive Rate: {:.2f}%".format(
    fp / (fp + tn) * 100
))
print("False Negative Rate: {:.2f}%".format(
    fn / (fn + tp) * 100
))
print("Missed Diabetes Cases:", fn)
print("Correctly Detected Diabetes Cases:", tp)
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))
print("Precision: {:.2f}%".format(
    precision_score(y_test, y_pred) * 100
))
print("Recall: {:.2f}%".format(
    recall_score(y_test, y_pred) * 100
))
print("F1 Score: {:.2f}%".format(
    f1_score(y_test, y_pred) * 100
))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
error_df = X_test.copy()
error_df["Actual"] = y_test
error_df["Predicted"] = y_pred
error_df["Prediction_Result"] = "Correct"
error_df.loc[
    (error_df["Actual"] == 1) &
    (error_df["Predicted"] == 0),
    "Prediction_Result"
] = "False Negative"
error_df.loc[
    (error_df["Actual"] == 0) &
    (error_df["Predicted"] == 1),
    "Prediction_Result"
] = "False Positive"
error_path = os.path.join(
    results_folder,
    "error_analysis.csv"
)
error_df.to_csv(error_path, index=False)
print("\nDetailed predictions saved to:", error_path)