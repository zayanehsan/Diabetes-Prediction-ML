import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]
print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions, zero_division=0))
print("Recall:", recall_score(y_test, logistic_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, logistic_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, logistic_probabilities))
print("PR-AUC:", average_precision_score(y_test, logistic_probabilities))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))