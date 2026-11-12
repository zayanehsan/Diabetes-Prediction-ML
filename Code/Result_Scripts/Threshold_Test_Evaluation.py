import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
threshold = 0.375
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
print("Training Gradient Boosting...")
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
y_pred = (probabilities >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
specificity = tn / (tn + fp)
print("\n===== TEST RESULTS AT THRESHOLD 0.375 =====")
print("Threshold:", threshold)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(specificity * 100))
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))
print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
test_results = pd.DataFrame([{
    "Threshold": threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Specificity": specificity,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])
output_path = os.path.join(
    results_folder,
    "threshold_test_results.csv"
)
test_results.to_csv(output_path, index=False)
print("\nResults saved to:", output_path)