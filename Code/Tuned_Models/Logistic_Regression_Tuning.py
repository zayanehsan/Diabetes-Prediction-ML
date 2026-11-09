import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
X_train = pd.read_csv("Processed_Data/X_train_raw.csv")
y_train = pd.read_csv("Processed_Data/Y_train_original.csv").squeeze()
X_test = pd.read_csv("Processed_Data/X_test_raw.csv")
y_test = pd.read_csv("Processed_Data/Y_test.csv").squeeze()
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("oversampler", RandomOverSampler(
        sampling_strategy=0.36,
        random_state=42
    )),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs"]
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
print("Starting Logistic Regression hyperparameter tuning...")
grid_search.fit(X_train, y_train)
print("\nBEST PARAMETERS:")
print(grid_search.best_params_)
print("\nBEST CROSS-VALIDATION F1 SCORE:")
print(grid_search.best_score_)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]
print("\nTUNED LOGISTIC REGRESSION TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("Average Precision:", average_precision_score(y_test, probabilities))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))