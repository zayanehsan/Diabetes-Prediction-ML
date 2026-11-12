import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
processed_folder = os.path.join(project_folder, "Processed_Data")
graph_folder = os.path.join(
    project_folder,
    "Graphs",
    "Confusion_Matrices"
)
os.makedirs(graph_folder, exist_ok=True)
X_train = pd.read_csv(os.path.join(processed_folder, "X_train.csv"))
y_train = pd.read_csv(os.path.join(processed_folder, "Y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(processed_folder, "X_test.csv"))
y_test = pd.read_csv(os.path.join(processed_folder, "Y_test.csv")).values.ravel()
X_train_original = pd.read_csv(os.path.join(processed_folder, "X_train_original.csv"))
y_train_original = pd.read_csv(os.path.join(processed_folder, "Y_train_original.csv")).values.ravel()
X_train_raw = pd.read_csv(os.path.join(processed_folder, "X_train_raw.csv"))
X_test_raw = pd.read_csv(os.path.join(processed_folder, "X_test_raw.csv"))
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
        max_depth=4,
        learning_rate=0.1,
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
        max_iter=1000,
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
def create_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("oversampler", RandomOverSampler(
            sampling_strategy=0.36,
            random_state=42
        )),
        ("model", model)
    ])
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")
print("\nTraining Baseline Model...")
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_original, y_train_original)
baseline_pred = baseline.predict(X_test)
plot_confusion_matrix(
    y_test,
    baseline_pred,
    "Baseline Confusion Matrix",
    os.path.join(graph_folder, "Baseline.png")
)
print("\nTraining Original Models...")
for name, model in original_models.items():
    print(f"Training: {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    filename = os.path.join(
        graph_folder,
        f"{name.replace(' ', '_')}_Original.png"
    )
    plot_confusion_matrix(
        y_test,
        predictions,
        f"{name} (Original)",
        filename
    )
print("\nTraining Tuned Models...")
for name, model in tuned_models.items():
    print(f"Training: {name}")
    pipeline = create_pipeline(model)
    pipeline.fit(X_train_raw, y_train_original)
    predictions = pipeline.predict(X_test_raw)
    filename = os.path.join(
        graph_folder,
        f"{name.replace(' ', '_')}_Tuned.png"
    )
    plot_confusion_matrix(
        y_test,
        predictions,
        f"{name} (Tuned)",
        filename
    )
print("\nAll confusion matrices generated successfully!")