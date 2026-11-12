import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_folder = os.path.join(project_folder, "Processed_Data")
results_folder = os.path.join(project_folder, "Results")
graph_folder = os.path.join(project_folder, "Graphs", "Feature_Importance")
os.makedirs(graph_folder, exist_ok=True)
X_train = pd.read_csv(os.path.join(data_folder, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(data_folder, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(data_folder, "Y_train_original.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(data_folder, "Y_test.csv")).values.ravel()
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
model.fit(X_train, y_train)
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": importance.importances_mean,
    "Importance_STD": importance.importances_std
})
importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
)
csv_path = os.path.join(results_folder, "feature_importance.csv")
importance_df.to_csv(csv_path, index=False)
print("\nPermutation Feature Importance:")
print(importance_df.to_string(index=False))
print("\nResults saved to:", csv_path)
plt.figure(figsize=(10, 8))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance_Mean"],
    xerr=importance_df["Importance_STD"]
)
plt.xlabel("Decrease in ROC-AUC")
plt.ylabel("Features")
plt.title("Feature Importance - Tuned Gradient Boosting")
plt.gca().invert_yaxis()
plt.tight_layout()
graph_path = os.path.join(
    graph_folder,
    "Gradient_Boosting_Feature_Importance.png"
)
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.show()
print("\nGraph saved to:", graph_path)