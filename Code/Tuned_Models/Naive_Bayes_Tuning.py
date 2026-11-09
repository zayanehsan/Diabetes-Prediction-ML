import pandas as pd
from sklearn.naive_bayes import GaussianNB
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
X_train = pd.read_csv("Processed_Data/X_train_raw.csv")
y_train = pd.read_csv("Processed_Data/Y_train_original.csv").squeeze()
X_test = pd.read_csv("Processed_Data/X_test_raw.csv")
y_test = pd.read_csv("Processed_Data/Y_test.csv").squeeze()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("oversampler", RandomOverSampler(
        sampling_strategy=0.36,
        random_state=42
    )),
    ("model", GaussianNB())
])
param_grid = {
    "model__var_smoothing": [
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7
    ]
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
print("Starting Naive Bayes Hyperparameter Tuning...")
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
print("\nNaive Bayes Tuned Model Results")
print("--------------------------------")
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