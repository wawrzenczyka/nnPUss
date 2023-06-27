# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.nnPUss.dataset import SCAR_SS_Labeler, Synthetic_PU_CC, Synthetic_PU_SS

c = 0.3
dataset = Synthetic_PU_CC(
    "data", SCAR_SS_Labeler(label_frequency=c), download=True, size=1_000
)


sns.set_theme()
plt.figure(figsize=(10, 6))

plt.scatter(
    dataset.data[:, 0],
    dataset.data[:, 1],
    s=3,
    c=np.where(dataset.pu_targets == 1, "r", "grey"),
    alpha=0.5,
)
plt.show()


# %%
