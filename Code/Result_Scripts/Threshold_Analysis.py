import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
print("Generating 5-fold cross-validation predictions...")
oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]
thresholds = np.arange(0.20, 0.501, 0.025)
results = []
for threshold in thresholds:
    predictions = (
        oof_probabilities >= threshold
    ).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_train,
        predictions,
        labels=[0, 1]
    ).ravel()
    results.append({
        "Threshold": round(threshold, 3),
        "Accuracy": accuracy_score(y_train, predictions),
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, predictions, zero_division=0
        ),
        "Specificity": tn / (tn + fp),
        "False_Negatives": fn,
        "False_Positives": fp
    })
results_df = pd.DataFrame(results)
print("\n===== THRESHOLD COMPARISON (5-FOLD CV) =====")
print(results_df.round(4).to_string(index=False))
output_path = os.path.join(
    results_folder,
    "threshold_analysis.csv"
)
results_df.to_csv(output_path, index=False)
print("\nResults saved to:", output_path)
best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]
print("\n===== HIGHEST CV F1 THRESHOLD =====")
print(best_f1_row.round(4))