import pandas as pd
import os
os.makedirs("Results", exist_ok=True)
results = [
    {
        "Model": "Logistic Regression",
        "Best_Parameters": "C=0.01, solver=lbfgs",
        "CV_F1": 0.419337,
        "Accuracy": 0.834261,
        "Precision": 0.463310,
        "Recall": 0.399373,
        "F1_Score": 0.428972,
        "ROC_AUC": 0.814560,
        "Average_Precision": 0.424425,
        "TN": 52146, "FP": 4871, "FN": 6324, "TP": 4205
    },
    {
        "Model": "Decision Tree",
        "Best_Parameters": "max_depth=5, min_samples_leaf=5, min_samples_split=10",
        "CV_F1": 0.433172,
        "Accuracy": 0.812483,
        "Precision": 0.413949,
        "Recall": 0.488176,
        "F1_Score": 0.448008,
        "ROC_AUC": 0.796302,
        "Average_Precision": 0.390196,
        "TN": 49740, "FP": 7277, "FN": 5389, "TP": 5140
    },
    {
        "Model": "Random Forest",
        "Best_Parameters": "n_estimators=100, max_depth=15, min_samples_leaf=10",
        "CV_F1": 0.421232,
        "Accuracy": 0.843218,
        "Precision": 0.496236,
        "Recall": 0.381898,
        "F1_Score": 0.431623,
        "ROC_AUC": 0.819922,
        "Average_Precision": 0.450075,
        "TN": 52935, "FP": 4082, "FN": 6508, "TP": 4021
    },
    {
        "Model": "Naive Bayes",
        "Best_Parameters": "var_smoothing=1e-11",
        "CV_F1": 0.431989,
        "Accuracy": 0.741732,
        "Precision": 0.330191,
        "Recall": 0.638617,
        "F1_Score": 0.435309,
        "ROC_AUC": 0.772112,
        "Average_Precision": 0.356776,
        "TN": 43377, "FP": 13640, "FN": 3805, "TP": 6724
    },
    {
        "Model": "SVM",
        "Best_Parameters": "C=0.01",
        "CV_F1": 0.408629,
        "Accuracy": 0.837622,
        "Precision": 0.473333,
        "Recall": 0.370026,
        "F1_Score": 0.415352,
        "ROC_AUC": 0.814807,
        "Average_Precision": 0.425509,
        "TN": 52682, "FP": 4335, "FN": 6633, "TP": 3896
    },
    {
        "Model": "Gradient Boosting",
        "Best_Parameters": "n_estimators=150, learning_rate=0.1, max_depth=3",
        "CV_F1": 0.447841,
        "Accuracy": 0.838495,
        "Precision": 0.480159,
        "Recall": 0.436699,
        "F1_Score": 0.457399,
        "ROC_AUC": 0.823265,
        "Average_Precision": 0.457932,
        "TN": 52039, "FP": 4978, "FN": 5931, "TP": 4598
    },
    {
        "Model": "XGBoost",
        "Best_Parameters": "n_estimators=200, max_depth=4, learning_rate=0.1",
        "CV_F1": 0.448177,
        "Accuracy": 0.838096,
        "Precision": 0.478693,
        "Recall": 0.434229,
        "F1_Score": 0.455378,
        "ROC_AUC": 0.822928,
        "Average_Precision": 0.457513,
        "TN": 52038, "FP": 4979, "FN": 5957, "TP": 4572
    }
]
df = pd.DataFrame(results)
df["Specificity"] = df["TN"] / (df["TN"] + df["FP"])
output_path = "Results/tuning_results.csv"
df.to_csv(output_path, index=False)
print("Tuning results saved successfully!")
print("\nTuning Results:")
print(df.round(4).to_string(index=False))
print("\nSaved at:", output_path)