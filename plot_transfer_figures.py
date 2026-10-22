
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/transfer_seed_summary_new_seeds.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)
weights = [0.6, 0.8]
labels = ["s = 0.6", "s = 0.8"]
x = [0, 1]

# Figure 2: Mean combined return with SD across training seeds.
metric = "sum_agent_environment_returns_mean"
groups = [df[df["s_interest"] == w][metric].dropna() for w in weights]
means = [g.mean() for g in groups]
stds = [g.std(ddof=1) for g in groups]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(
    x, means, yerr=stds, fmt="o",
    capsize=7, markersize=8, linewidth=1.5
)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xticks(x, labels)
ax.set_xlabel("Reward-mixing weight")
ax.set_ylabel("Mean combined environmental return")
ax.set_title("Combined return across training seeds")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "transfer_mean_return_errorbars.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 5: Paired difference for each training seed.
wide = df.pivot(
    index="training_seed",
    columns="s_interest",
    values=metric
).dropna(subset=weights)

wide["difference"] = wide[0.8] - wide[0.6]

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["tab:blue" if v >= 0 else "tab:orange"
          for v in wide["difference"]]

ax.bar(
    wide.index.astype(str),
    wide["difference"],
    color=colors
)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Training seed")
ax.set_ylabel("Return difference: s=0.8 minus s=0.6")
ax.set_title("Paired change in combined return")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "transfer_paired_differences.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 4: Individual agent environmental returns.
fig, ax = plt.subplots(figsize=(6.5, 4.5))

agent_metrics = [
    ("agent0_environment_return_mean", "Agent 0"),
    ("agent1_environment_return_mean", "Agent 1"),
]

offsets = [-0.08, 0.08]

for (column, label), offset in zip(agent_metrics, offsets):
    means = [
        df.loc[df["s_interest"] == w, column].mean()
        for w in weights
    ]
    stds = [
        df.loc[df["s_interest"] == w, column].std(ddof=1)
        for w in weights
    ]
    ax.errorbar(
        [v + offset for v in x],
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        label=label,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x, labels)
ax.set_xlabel("Reward-mixing weight")
ax.set_ylabel("Mean individual environmental return")
ax.set_title("Individual-agent outcomes across training seeds")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "transfer_individual_agent_returns.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

print("Created:")
for name in [
    "transfer_mean_return_errorbars.png",
    "transfer_paired_differences.png",
    "transfer_individual_agent_returns.png",
]:
    print((OUT / name).resolve())

print("\nMean combined returns:")
for w in weights:
    values = df.loc[df["s_interest"] == w, metric]
    print(f"s={w}: mean={values.mean():.3f}, "
          f"SD across seeds={values.std(ddof=1):.3f}")

print("Mean paired difference (0.8 - 0.6):",
      round(wide["difference"].mean(), 3))
