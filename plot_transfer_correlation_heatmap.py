
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("results/transfer_evaluation_new_seeds.csv")
OUTPUT = Path("heatmap_final.png")

df = pd.read_csv(INPUT)
df["s_interest"] = df["trained_policy"].str.extract(
    r"s(0\.\d+)_seed"
).astype(float)

metrics = {
    "agent0_environment_return": "Agent 0 return",
    "agent1_environment_return": "Agent 1 return",
    "agent0_eat_own_coins_metric_sum": "Agent 0 own-coin metric",
    "agent1_eat_own_coins_metric_sum": "Agent 1 own-coin metric",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

for ax, weight in zip(axes, [0.6, 0.8]):
    subset = df[df["s_interest"] == weight]
    corr = subset[list(metrics)].corr()

    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    labels = list(metrics.values())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_title(f"s = {weight}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=9,
                color="black",
            )

cbar = fig.colorbar(
    image,
    ax=axes,
    label="Pearson correlation (r)",
    shrink=0.85,
    pad=0.04,
)

fig.suptitle("Correlation between evaluation metrics", fontsize=14)


fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT.resolve())
print("Constant episode-length metric excluded.")
print("Correlations are descriptive; episodes are clustered by checkpoint.")
