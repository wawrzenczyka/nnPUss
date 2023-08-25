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
pd.set_option("display.max_rows", 500)
(
    results_df[results_df.dataset.str.contains(" SS")].pivot_table(
        values="accuracy",
        index=["dataset", "label_frequency"],
        columns="model",
        aggfunc=len,
    )
)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains(" CC")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUcc improvement"] = df["nnPUcc"] - df["nnPUss"]
    # df["uPUcc improvement"] = df["uPUcc"] - df["uPUss"]
    df = df.round(2)
    df.to_csv(f"csv/CC-datasets-{metric}.csv")
    diff_pivot = df.reset_index(drop=False).pivot_table(
        values="nnPUcc improvement",
        index="label_frequency",
        columns="dataset",
    )
    diff_pivot.to_csv(f"csv/CC-datasets-{metric}-diff-pivot.csv")
    if metric in ["accuracy"]:
        display(df)
        display(diff_pivot)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains(" SS")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUss improvement"] = df["nnPUss"] - df["nnPUcc"]
    # df["uPUss improvement"] = df["uPUss"] - df["uPUcc"]
    df = df.round(2)
    df.to_csv(f"csv/SS-datasets-{metric}.csv")
    diff_pivot = df.reset_index(drop=False).pivot_table(
        values="nnPUss improvement",
        index="label_frequency",
        columns="dataset",
    )
    diff_pivot.to_csv(f"csv/SS-datasets-{metric}-diff-pivot.csv")
    if metric in ["accuracy"]:
        display(df)
        display(diff_pivot)

# %%
results_df
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = results_df[results_df.dataset.str.contains(" CC")].pivot_table(
        values=metric, index=["dataset", "label_frequency"], columns="model"
    )
    df["Improvement"] = df["nnPUcc"] - df["nnPUss"]
    df = df.reset_index(drop=False).melt(
        id_vars=["dataset", "label_frequency"], value_name=metric
    )
    df.model = pd.Categorical(
        df.model, categories=["nnPUss", "nnPUcc", "Improvement"], ordered=True
    )
    df = (
        df.pivot_table(
            values=metric, index=["label_frequency", "model"], columns=["dataset"]
        )
        * 100
    )
    # df["uPUcc improvement"] = df["uPUcc"] - df["uPUss"]
    df = df.round(2)
    df.to_csv(f"csv/CC-datasets-{metric}.csv")
    # diff_pivot = df.reset_index(drop=False).pivot_table(
    #     values="nnPUcc improvement",
    #     index="label_frequency",
    #     columns="dataset",
    # )
    # diff_pivot.to_csv(f"csv/CC-datasets-{metric}-diff-pivot.csv")
    if metric in ["accuracy"]:
        display(df)
        # display(diff_pivot)

    with open("test.tex", "w") as f:
        df.to_latex(f)

# %%
results_df[
    (results_df.dataset.str.contains("CC"))
    & (results_df.model == "nnPUss")
    & (results_df.label_frequency == 0.1)
]

# %%
sizes = {
    "[IMG] Beans SS": 1034,
    "[IMG] CIFAR SS": 60000,
    "[IMG] Chest X-ray SS": 4077,
    "[IMG] DogFood SS": 2500,
    "[IMG] EuroSAT SS": 21600,
    "[IMG] FashionMNIST SS": 60000,
    "[IMG] MNIST SS": 60000,
    "[IMG] Oxford Pets SS": 7390,
    "[IMG] Snacks SS": 4838,
    "[TAB] California SS": 20634,
    "[TAB] Credit SS": 16714,
    "[TAB] Electricity SS": 38474,
    "[TAB] Wine SS": 2554,
    "[TXT] 20News SS": 11300,
    "[TXT] HateSpeech SS": 10900,
    "[TXT] IMDB SS": 25000,
    "[TXT] PoemSentiment SS": 892,
    "[TXT] SMSSpam SS": 5574,
}
