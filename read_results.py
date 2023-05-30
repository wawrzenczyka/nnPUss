# %%
import json
import os

import pandas as pd
from IPython.display import display

RESULTS_DIR = "output"

results = []
for directory, subdirs, files in os.walk(RESULTS_DIR):
    if "metrics.json" in files:
        metrics_file_path = os.path.join(directory, "metrics.json")
        with open(metrics_file_path, "r") as f:
            metric_values = json.load(f)
            results.append(metric_values)
        # print(metrics_file_path)

results_df = pd.DataFrame.from_records(results)
display(results_df)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains("CC")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUcc improvement"] = df["nnPUcc"] - df["nnPUss"]
    df["uPUcc improvement"] = df["uPUcc"] - df["uPUss"]
    df = df.round(2)
    df.to_csv(f"csv/{metric}-nnPU-CC-datasets.csv")
    if metric in ["accuracy"]:
        display(df)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains("SS")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUss improvement"] = df["nnPUss"] - df["nnPUcc"]
    df["uPUss improvement"] = df["uPUss"] - df["uPUcc"]
    df = df.round(2)
    df.to_csv(f"csv/{metric}-nnPU-SS-datasets.csv")
    if metric in ["accuracy"]:
        display(df)


# %%
