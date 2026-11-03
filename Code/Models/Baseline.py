import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
X_train = pd.read_csv("Processed_Data/X_train_original.csv")
y_train = pd.read_csv("Processed_Data/Y_train_original.csv").squeeze()
X_test = pd.read_csv("Processed_Data/X_test.csv")
y_test = pd.read_csv("Processed_Data/Y_test.csv").squeeze()
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)
print("\nBASELINE MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print("Precision:", precision_score(y_test, baseline_predictions, zero_division=0))
print("Recall:", recall_score(y_test, baseline_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, baseline_predictions, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, baseline_predictions))