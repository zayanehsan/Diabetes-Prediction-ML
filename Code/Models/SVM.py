import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
project_folder = Path(__file__).resolve().parents[2]
processed_folder = project_folder / "Processed_Data"
X_train = pd.read_csv(processed_folder / "X_train.csv")
y_train = pd.read_csv(processed_folder / "Y_train.csv").squeeze()
X_test = pd.read_csv(processed_folder / "X_test.csv")
y_test = pd.read_csv(processed_folder / "Y_test.csv").squeeze()
svm_model = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42
)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_scores = svm_model.decision_function(X_test)
print("\nSUPPORT VECTOR MACHINE RESULTS")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Precision:", precision_score(y_test, svm_predictions, zero_division=0))
print("Recall:", recall_score(y_test, svm_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, svm_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, svm_scores))
print("PR-AUC:", average_precision_score(y_test, svm_scores))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))