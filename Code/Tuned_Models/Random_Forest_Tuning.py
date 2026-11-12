import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
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
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 15],
    "model__min_samples_leaf": [5, 10]
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
print("Starting Random Forest hyperparameter tuning...")
grid_search.fit(X_train, y_train)
print("\nBEST PARAMETERS:")
print(grid_search.best_params_)
print("\nBEST CROSS-VALIDATION F1 SCORE:")
print(grid_search.best_score_)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]
print("\nTUNED RANDOM FOREST TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("Average Precision:", average_precision_score(y_test, probabilities))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))