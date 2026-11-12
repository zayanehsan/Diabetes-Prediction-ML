import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    auc
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
results_folder = os.path.join(project_folder, "Results")
graph_folder = os.path.join(
    project_folder,
    "Graphs",
    "ROC_PR"
)
os.makedirs(graph_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
X_train = pd.read_csv(os.path.join(project_folder, "Processed_Data", "X_train.csv"))
y_train = pd.read_csv(os.path.join(project_folder, "Processed_Data", "Y_train.csv")).squeeze()
X_train_raw = pd.read_csv(os.path.join(project_folder, "Processed_Data", "X_train_raw.csv"))
y_train_original = pd.read_csv(os.path.join(project_folder, "Processed_Data", "Y_train_original.csv")).squeeze()
X_test = pd.read_csv(os.path.join(project_folder, "Processed_Data", "X_test.csv"))
X_test_raw = pd.read_csv(os.path.join(project_folder, "Processed_Data", "X_test_raw.csv"))
y_test = pd.read_csv(os.path.join(project_folder, "Processed_Data", "Y_test.csv")).squeeze()
original_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(),
    "SVM": LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
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
}
tuned_models = {
    "Logistic Regression": LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(
        var_smoothing=1e-11
    ),
    "SVM": LinearSVC(
        C=0.01,
        max_iter=4000,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
}
all_scores = {}
metrics_results = []
def get_prediction_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
print("\nTraining Original Models...")
for name, model in original_models.items():
    print("Training:", name)
    model.fit(X_train, y_train)
    scores = get_prediction_scores(model, X_test)
    all_scores[name + " (Original)"] = scores
print("\nTraining Tuned Models...")
for name, model in tuned_models.items():
    print("Training:", name)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("oversampler", RandomOverSampler(
            sampling_strategy=0.36,
            random_state=42
        )),
        ("model", model)
    ])
    pipeline.fit(X_train_raw, y_train_original)
    scores = get_prediction_scores(pipeline, X_test_raw)
    all_scores[name + " (Tuned)"] = scores
for model_name, scores in all_scores.items():
    roc_auc = roc_auc_score(y_test, scores)
    average_precision = average_precision_score(
        y_test,
        scores
    )
    precision, recall, _ = precision_recall_curve(
        y_test,
        scores
    )
    exact_pr_auc = auc(recall, precision)
    metrics_results.append({
        "Model": model_name,
        "ROC_AUC": roc_auc,
        "Average_Precision": average_precision,
        "Exact_PR_AUC": exact_pr_auc
    })
for version in ["Original", "Tuned"]:
    plt.figure(figsize=(11, 7))
    for model_name, scores in all_scores.items():
        if version not in model_name:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )
    plt.title(f"ROC Curves - {version} Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    filename = os.path.join(
        graph_folder,
        f"ROC_Curve_{version}.png"
    )
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Saved:", filename)
for version in ["Original", "Tuned"]:
    plt.figure(figsize=(11, 7))
    for model_name, scores in all_scores.items():
        if version not in model_name:
            continue
        precision, recall, _ = precision_recall_curve(
            y_test,
            scores
        )
        average_precision = average_precision_score(
            y_test,
            scores
        )
        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AP = {average_precision:.3f})"
        )
    baseline_prevalence = y_test.mean()
    plt.axhline(
        y=baseline_prevalence,
        linestyle="--",
        label="No-Skill Baseline"
    )
    plt.title(f"Precision-Recall Curves - {version} Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()
    filename = os.path.join(
        graph_folder,
        f"PR_Curve_{version}.png"
    )
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Saved:", filename)
metrics_df = pd.DataFrame(metrics_results)
output_path = os.path.join(
    results_folder,
    "ROC_PR_Metrics.csv"
)
metrics_df.to_csv(output_path, index=False)
print("\nROC and Precision-Recall Metrics:")
print(metrics_df.round(5).to_string(index=False))
print("\nMetrics saved at:", output_path)
print("\nAll ROC and Precision-Recall graphs generated successfully!")