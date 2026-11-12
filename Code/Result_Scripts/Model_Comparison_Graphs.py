import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
results_folder = os.path.join(project_folder, "Results")
graph_folder = os.path.join(
    project_folder,
    "Graphs",
    "Model_Comparison"
)
os.makedirs(graph_folder, exist_ok=True)
df = pd.read_csv(
    os.path.join(results_folder, "final_model_comparison.csv")
)
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "Average_Precision"
]
sns.set_theme(style="whitegrid")
for metric in metrics:
    original_column = metric + "_Original"
    tuned_column = metric + "_Tuned"
    plot_df = df[
        ["Model", original_column, tuned_column]
    ]
    plot_df = plot_df.melt(
        id_vars="Model",
        var_name="Version",
        value_name="Score"
    )
    plot_df["Version"] = plot_df["Version"].replace({
        original_column: "Original",
        tuned_column: "Tuned"
    })
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Version"
    )
    plt.title(
        f"Original vs Tuned Models - {metric.replace('_', ' ')}",
        fontsize=14
    )
    plt.xlabel("Machine Learning Models")
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model Version")
    plt.tight_layout()
    filename = os.path.join(
        graph_folder,
        f"{metric}_Comparison.png"
    )
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {filename}")
print("\nAll model comparison graphs generated successfully!")