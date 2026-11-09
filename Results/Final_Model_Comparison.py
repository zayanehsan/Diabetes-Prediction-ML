import pandas as pd
import os
os.makedirs("Results", exist_ok=True)
original_df = pd.read_csv("Results/model_metrics.csv")
tuned_df = pd.read_csv("Results/tuning_results.csv")
print("Original Model Results:")
print(original_df[["Model", "F1_Score"]])
print("\nTuned Model Results:")
print(tuned_df[["Model", "F1_Score"]])
comparison_df = pd.merge(
    original_df,
    tuned_df,
    on="Model",
    suffixes=("_Original", "_Tuned")
)
print("\nOriginal vs Tuned Model Comparison:")
print(
    comparison_df[
        ["Model", "F1_Score_Original", "F1_Score_Tuned"]
    ].round(4).to_string(index=False)
)
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "Average_Precision"
]
for metric in metrics:
    comparison_df[metric + "_Improvement"] = (
        comparison_df[metric + "_Tuned"] -
        comparison_df[metric + "_Original"]
    )
improvement_columns = (
    ["Model"] +
    [metric + "_Improvement" for metric in metrics]
)
print("\nImprovement After Hyperparameter Tuning:")
print(
    comparison_df[improvement_columns]
    .round(5)
    .to_string(index=False)
)
output_path = "Results/final_model_comparison.csv"
comparison_df.to_csv(output_path, index=False)
print("\nFinal model comparison saved successfully!")
print("Saved at:", output_path)
