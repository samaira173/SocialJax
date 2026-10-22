
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results")
OUT.mkdir(exist_ok=True)

EVAL_FILE = OUT / "transfer_evaluation_extended.csv"
WEIGHTS = [0.6, 0.8]
SEEDS = range(142, 148)

# --------------------------------------------------
# 1. Load and validate evaluation data
# --------------------------------------------------
if not EVAL_FILE.exists():
    raise FileNotFoundError(f"Missing evaluation file: {EVAL_FILE}")

ev = pd.read_csv(EVAL_FILE)

required = {
    "trained_policy",
    "episode",
    "episode_seed",
    "agent0_environment_return",
    "agent1_environment_return",
    "sum_agent_environment_returns",
}
missing = required - set(ev.columns)
if missing:
    raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")

ev["weight"] = ev["trained_policy"].str.extract(
    r"transfer_s0[._](\d+)_seed"
)[0].astype(float) / 10

ev["training_seed"] = ev["trained_policy"].str.extract(
    r"seed(\d+)"
)[0].astype(int)

if set(ev["weight"].unique()) != set(WEIGHTS):
    raise ValueError(f"Unexpected weights: {sorted(ev['weight'].unique())}")

counts = ev.groupby(["weight", "training_seed"]).size()
if len(counts) != 12 or not (counts == 10).all():
    raise ValueError(
        "Expected 10 evaluation episodes for each of 12 trained policies."
    )

# Save a normalized copy with explicit weight and training-seed columns.
ev.to_csv(OUT / "extended_evaluation_with_conditions.csv", index=False)

# --------------------------------------------------
# 2. Per-policy results: average over its 10 episodes
# --------------------------------------------------
policy = (
    ev.groupby(["weight", "training_seed"], as_index=False)
    .agg(
        mean_combined_return=("sum_agent_environment_returns", "mean"),
        sd_combined_return=("sum_agent_environment_returns", "std"),
        mean_agent0_return=("agent0_environment_return", "mean"),
        mean_agent1_return=("agent1_environment_return", "mean"),
        mean_episode_length=("steps", "mean") if "steps" in ev.columns
        else ("episode", "count"),
        episodes=("episode", "count"),
    )
)

policy.to_csv(OUT / "extended_policy_summary.csv", index=False)

# Across-seed statistics: each independent training run counts once.
summary = (
    policy.groupby("weight")
    .agg(
        mean_combined_return=("mean_combined_return", "mean"),
        sd_across_training_seeds=("mean_combined_return", "std"),
        mean_agent0_return=("mean_agent0_return", "mean"),
        sd_agent0_across_seeds=("mean_agent0_return", "std"),
        mean_agent1_return=("mean_agent1_return", "mean"),
        sd_agent1_across_seeds=("mean_agent1_return", "std"),
        number_of_training_seeds=("training_seed", "nunique"),
    )
    .reset_index()
)
summary.to_csv(OUT / "extended_condition_summary.csv", index=False)

# Paired comparison: same training seed across both weights.
wide = policy.pivot(
    index="training_seed",
    columns="weight",
    values="mean_combined_return",
)
wide = wide.rename(columns={0.6: "return_s06", 0.8: "return_s08"})
wide["difference_s08_minus_s06"] = wide["return_s08"] - wide["return_s06"]
wide.to_csv(OUT / "extended_paired_seed_comparison.csv")

# Paired-difference summary.
diff = wide["difference_s08_minus_s06"]
diff_summary = pd.DataFrame([{
    "mean_paired_difference_s08_minus_s06": diff.mean(),
    "sd_paired_differences": diff.std(ddof=1),
    "se_paired_difference": diff.std(ddof=1) / np.sqrt(len(diff)),
    "seeds_s08_higher": int((diff > 0).sum()),
    "seeds_s06_higher": int((diff < 0).sum()),
    "ties": int((diff == 0).sum()),
    "number_of_pairs": len(diff),
}])
diff_summary.to_csv(OUT / "extended_paired_difference_summary.csv", index=False)

# --------------------------------------------------
# Figure 1: training curves from the extended CSVs
# --------------------------------------------------
frames = []
for weight in WEIGHTS:
    for seed in SEEDS:
        path = OUT / f"transfer_coin_s{weight}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing training CSV: {path}")
        df = pd.read_csv(path)
        for col in ["env_step", "original_rewards"]:
            if col not in df.columns:
                raise ValueError(f"{path} missing column {col}")
        df["weight"] = weight
        df["training_seed"] = seed
        frames.append(df)

train = pd.concat(frames, ignore_index=True)
train_summary = (
    train.groupby(["weight", "env_step"])["original_rewards"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
train_summary.to_csv(OUT / "extended_learning_curve_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for weight in WEIGHTS:
    g = train_summary[train_summary["weight"] == weight].sort_values("env_step")
    x = g["env_step"].to_numpy()
    mean = g["mean"].to_numpy()
    sd = g["std"].fillna(0).to_numpy()
    ax.plot(x, mean, marker="o", markersize=3, linewidth=1.8,
            label=f"s_interest = {weight}")
    ax.fill_between(x, mean - sd, mean + sd, alpha=0.18)

ax.set_xlim(0, 32768)
ax.set_xlabel("Environment steps")
ax.set_ylabel("Mean logged original reward per training update")
ax.set_title("Training reward metric across six seeds")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "extended_training_learning_curves.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# Figure 2: mean final evaluation return, SD across seeds
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
means = summary["mean_combined_return"].to_numpy()
errs = summary["sd_across_training_seeds"].to_numpy()
labels = [f"s_interest = {w}" for w in summary["weight"]]
ax.bar(labels, means, yerr=errs, capsize=6)
ax.axhline(0, linewidth=1)
ax.set_ylabel("Mean combined individual reward per episode")
ax.set_title("Final policy performance (SD across training seeds)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "extended_mean_return_errorbars.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# Figure 3: paired comparison by training seed
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for seed, row in wide.iterrows():
    axes[0].plot(
        [0, 1],
        [row["return_s06"], row["return_s08"]],
        marker="o",
        alpha=0.8,
        linewidth=1.3,
        label=f"Seed {seed}",
    )
axes[0].set_xticks([0, 1], ["s_interest = 0.6", "s_interest = 0.8"])
axes[0].set_ylabel("Mean combined evaluation return")
axes[0].set_title("Matched training seeds")
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    wide.index.astype(str),
    wide["difference_s08_minus_s06"],
)
axes[1].axhline(0, linewidth=1)
axes[1].set_xlabel("Training seed")
axes[1].set_ylabel("Return difference (0.8 − 0.6)")
axes[1].set_title("Paired differences")
axes[1].grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "extended_paired_seed_comparison.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# Figure 4: individual agent outcomes
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for weight in WEIGHTS:
    g = ev[ev["weight"] == weight]
    ax.scatter(
        g["agent0_environment_return"],
        g["agent1_environment_return"],
        alpha=0.65,
        s=28,
        label=f"s_interest = {weight}",
    )

low = min(
    ev["agent0_environment_return"].min(),
    ev["agent1_environment_return"].min(),
)
high = max(
    ev["agent0_environment_return"].max(),
    ev["agent1_environment_return"].max(),
)
ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
ax.set_xlabel("Agent 0 evaluation return")
ax.set_ylabel("Agent 1 evaluation return")
ax.set_title("Individual-agent outcomes across evaluation episodes")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "extended_agent_return_scatter.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# Optional: descriptive correlation heatmaps
# --------------------------------------------------
metric_cols = [
    "agent0_environment_return",
    "agent1_environment_return",
    "agent0_eat_own_coins_metric_sum",
    "agent1_eat_own_coins_metric_sum",
]

short_labels = [
    "Agent 0 return",
    "Agent 1 return",
    "Agent 0 own-coin metric",
    "Agent 1 own-coin metric",
]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

image = None

for ax, weight in zip(axes, WEIGHTS):
    g = ev[ev["weight"] == weight]
    corr = g[metric_cols].corr()

    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="equal",
    )

    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=9)

    ax.set_title(f"s_interest = {weight}", fontsize=12, pad=12)

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            value = corr.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color="black",
            )

fig.colorbar(
    image,
    ax=axes,
    shrink=0.78,
    pad=0.04,
    label="Pearson correlation",
)

fig.suptitle(
    "Exploratory correlations between evaluation metrics",
    fontsize=14,
)

fig.savefig(
    OUT / "extended_correlation_heatmaps.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# Print key findings
# --------------------------------------------------
print("\nEVALUATION SUMMARY (SD is across six training seeds)")
print(summary.round(3).to_string(index=False))

print("\nPAIRED DIFFERENCE SUMMARY (s_interest 0.8 minus 0.6)")
print(diff_summary.round(3).to_string(index=False))

print("\nSaved main figures:")
for name in [
    "extended_training_learning_curves.png",
    "extended_mean_return_errorbars.png",
    "extended_paired_seed_comparison.png",
    "extended_agent_return_scatter.png",
    "extended_correlation_heatmaps.png",
]:
    print(" ", (OUT / name).resolve())

print("\nSaved summary CSVs:")
for name in [
    "extended_policy_summary.csv",
    "extended_condition_summary.csv",
    "extended_paired_seed_comparison.csv",
    "extended_paired_difference_summary.csv",
    "extended_learning_curve_summary.csv",
]:
    print(" ", (OUT / name).resolve())
