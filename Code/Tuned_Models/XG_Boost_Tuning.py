import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
project_folder = Path(__file__).resolve().parents[2]
processed_folder = project_folder / "Processed_Data"
X_train = pd.read_csv(processed_folder / "X_train_raw.csv")
y_train = pd.read_csv(processed_folder / "Y_train_original.csv").squeeze()
X_test = pd.read_csv(processed_folder / "X_test_raw.csv")
y_test = pd.read_csv(processed_folder / "Y_test.csv").squeeze()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("oversampler", RandomOverSampler(
        sampling_strategy=0.36,
        random_state=42
    )),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 4],
    "model__learning_rate": [0.05, 0.1]
}
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=2,
    verbose=2
)
print("Starting XGBoost Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)
print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation F1 Score:")
print(grid_search.best_score_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_scores = best_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\nTuned XGBoost Results")
print("---------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
print("Average Precision:", average_precision)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
print("\nXGBoost model parameters:")
print(best_model.get_params())