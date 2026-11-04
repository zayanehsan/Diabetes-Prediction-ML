import pandas as pd
import os
os.makedirs("Results", exist_ok=True)
results = [
    {
        "Model": "Baseline",
        "Accuracy": 0.844121,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1_Score": 0.0,
        "ROC_AUC": None,
        "Average_Precision": None,
        "TN": 57017, "FP": 0, "FN": 10529, "TP": 0
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.834039,
        "Precision": 0.462644,
        "Recall": 0.400513,
        "F1_Score": 0.429342,
        "ROC_AUC": 0.814546,
        "Average_Precision": 0.424356,
        "TN": 52119, "FP": 4898, "FN": 6312, "TP": 4217
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.829168,
        "Precision": 0.448469,
        "Recall": 0.417419,
        "F1_Score": 0.432387,
        "ROC_AUC": 0.800596,
        "Average_Precision": 0.412784,
        "TN": 51612, "FP": 5405, "FN": 6134, "TP": 4395
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.842700,
        "Precision": 0.494108,
        "Recall": 0.382278,
        "F1_Score": 0.431058,
        "ROC_AUC": 0.819564,
        "Average_Precision": 0.448472,
        "TN": 52896, "FP": 4121, "FN": 6504, "TP": 4025
    },
    {
        "Model": "Naive Bayes",
        "Accuracy": 0.741761,
        "Precision": 0.330240,
        "Recall": 0.638712,
        "F1_Score": 0.435374,
        "ROC_AUC": 0.772117,
        "Average_Precision": 0.356773,
        "TN": 43378, "FP": 13639, "FN": 3804, "TP": 6725
    },
    {
        "Model": "SVM",
        "Accuracy": 0.837563,
        "Precision": 0.473109,
        "Recall": 0.370121,
        "F1_Score": 0.415326,
        "ROC_AUC": 0.814807,
        "Average_Precision": 0.425510,
        "TN": 52677, "FP": 4340, "FN": 6632, "TP": 3897
    },
    {
        "Model": "Gradient Boosting",
        "Accuracy": 0.839132,
        "Precision": 0.482092,
        "Recall": 0.430810,
        "F1_Score": 0.455011,
        "ROC_AUC": 0.823043,
        "Average_Precision": 0.457275,
        "TN": 52144, "FP": 4873, "FN": 5993, "TP": 4536
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.838421,
        "Precision": 0.479794,
        "Recall": 0.434134,
        "F1_Score": 0.455824,
        "ROC_AUC": 0.823259,
        "Average_Precision": 0.457637,
        "TN": 52061, "FP": 4956, "FN": 5958, "TP": 4571
    }
]
df = pd.DataFrame(results)
df["Specificity"] = df["TN"] / (df["TN"] + df["FP"])
output_path = "Results/model_metrics.csv"
df.to_csv(output_path, index=False)
print("Model comparison results saved successfully!")
print("\nModel Performance:")
print(df.round(4).to_string(index=False))
print("\nSaved at:", output_path)