import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
X_train = pd.read_csv("Processed_Data/X_train.csv")
y_train = pd.read_csv("Processed_Data/Y_train.csv").squeeze()
X_test = pd.read_csv("Processed_Data/X_test.csv")
y_test = pd.read_csv("Processed_Data/Y_test.csv").squeeze()
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2
)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]
print("\nXGBOOST RESULTS")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Precision:", precision_score(y_test, xgb_predictions, zero_division=0))
print("Recall:", recall_score(y_test, xgb_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, xgb_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probabilities))
print("PR-AUC:", average_precision_score(y_test, xgb_probabilities))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))