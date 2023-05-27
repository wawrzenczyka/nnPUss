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
(
    results_df[results_df.dataset.str.contains("CC")].pivot_table(
        values="accuracy", index=["dataset", "label_frequency"], columns="model"
    )
    * 100
).round(2)

# %%
(
    results_df[results_df.dataset.str.contains("SS")].pivot_table(
        values="accuracy", index=["dataset", "label_frequency"], columns="model"
    )
    * 100
).round(2)

# %%
