
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/transfer_seed_summary_new_seeds.csv")
OUTPUT = Path("results/transfer_paired_comparison.png")

df = pd.read_csv(INPUT)

# One mean combined return per trained checkpoint.
metric = "sum_agent_environment_returns_mean"

wide = df.pivot(
    index="training_seed",
    columns="s_interest",
    values=metric,
).dropna(subset=[0.6, 0.8])

fig, ax = plt.subplots(figsize=(7, 5))

# Connect the two conditions for each matched training seed.
for seed, row in wide.iterrows():
    ax.plot(
        [0, 1],
        [row[0.6], row[0.8]],
        marker="o",
        linewidth=1.3,
        alpha=0.65,
        label=f"Seed {seed}",
    )

# Overlay the mean across the six training seeds.
means = [wide[0.6].mean(), wide[0.8].mean()]
ax.scatter(
    [0, 1],
    means,
    marker="D",
    s=110,
    zorder=5,
    label="Mean across seeds",
)

ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xticks([0, 1], ["s = 0.6", "s = 0.8"])
ax.set_ylabel("Mean combined environmental return")
ax.set_xlabel("Reward-mixing weight")
ax.set_title("Paired comparison across training seeds")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT.resolve())
print("Mean return at s=0.6:", round(wide[0.6].mean(), 3))
print("Mean return at s=0.8:", round(wide[0.8].mean(), 3))
print("Paired difference (0.8 - 0.6):",
      round((wide[0.8] - wide[0.6]).mean(), 3))
