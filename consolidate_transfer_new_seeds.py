
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FILES = [
    Path("results/transfer_evaluation_new_seeds.csv"),
]

OUT_DIR = Path("results")
DETAIL_OUT = OUT_DIR / "all_transfer_evaluations_new_seeds.csv"
SEED_OUT = OUT_DIR / "transfer_seed_summary_new_seeds.csv"
PAIR_OUT = OUT_DIR / "paired_seed_comparison_new_seeds.csv"
GRAPH_OUT = OUT_DIR / "transfer_results_new_seeds.png"

for path in FILES:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

df = pd.concat([pd.read_csv(path) for path in FILES], ignore_index=True)

# Derive the reward weight and training seed from the policy label.
df["s_interest"] = df["trained_policy"].str.extract(r"s(0\.\d+)_seed")[0].astype(float)
df["training_seed"] = df["trained_policy"].str.extract(r"seed(\d+)")[0].astype(int)

df.to_csv(DETAIL_OUT, index=False)

metrics = [
    "agent0_environment_return",
    "agent1_environment_return",
    "sum_agent_environment_returns",
    "agent0_eat_own_coins_metric_sum",
    "agent1_eat_own_coins_metric_sum",
]

# Average the ten evaluation episodes for each trained checkpoint.
seed_summary = (
    df.groupby(["s_interest", "training_seed"])[metrics]
    .agg(["mean", "std"])
)
seed_summary.columns = [
    f"{metric}_{stat}" for metric, stat in seed_summary.columns
]
seed_summary = seed_summary.reset_index()
seed_summary.to_csv(SEED_OUT, index=False)

# Direct paired comparisons where both weights used the same training seed.
means = (
    df.groupby(["s_interest", "training_seed"])[metrics]
    .mean()
    .reset_index()
)

paired_rows = []
for seed in sorted(means["training_seed"].unique()):
    subset = means[means["training_seed"] == seed]
    if set(subset["s_interest"]) >= {0.6, 0.8}:
        low = subset[subset["s_interest"] == 0.6].iloc[0]
        high = subset[subset["s_interest"] == 0.8].iloc[0]
        paired_rows.append({
            "training_seed": seed,
            "mean_combined_return_s0.6": low["sum_agent_environment_returns"],
            "mean_combined_return_s0.8": high["sum_agent_environment_returns"],
            "difference_s0.8_minus_s0.6": (
                high["sum_agent_environment_returns"]
                - low["sum_agent_environment_returns"]
            ),
            "mean_agent0_return_s0.6": low["agent0_environment_return"],
            "mean_agent0_return_s0.8": high["agent0_environment_return"],
            "mean_agent1_return_s0.6": low["agent1_environment_return"],
            "mean_agent1_return_s0.8": high["agent1_environment_return"],
        })

paired = pd.DataFrame(paired_rows)
paired.to_csv(PAIR_OUT, index=False)

print("\nMEAN RESULTS PER TRAINING CHECKPOINT")
print(seed_summary[
    ["s_interest", "training_seed",
     "sum_agent_environment_returns_mean",
     "sum_agent_environment_returns_std",
     "agent0_environment_return_mean",
     "agent1_environment_return_mean"]
].round(2).to_string(index=False))

print("\nPAIRED TRAINING-SEED COMPARISON (0.8 minus 0.6)")
print(paired.round(2).to_string(index=False))

if not paired.empty:
    print(
        "\nAverage paired difference in combined return:",
        round(paired["difference_s0.8_minus_s0.6"].mean(), 2)
    )

# Plot the mean combined return per trained checkpoint.
fig, ax = plt.subplots(figsize=(9, 5))

for weight in [0.6, 0.8]:
    subset = seed_summary[seed_summary["s_interest"] == weight].sort_values("training_seed")
    ax.plot(
        subset["training_seed"],
        subset["sum_agent_environment_returns_mean"],
        marker="o",
        label=f"s = {weight}",
    )

ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Training seed")
ax.set_ylabel("Mean combined environmental return")
ax.set_title("TRANSFER: mean evaluation return by training seed (new runs)")
ax.set_xticks(sorted(seed_summary["training_seed"].unique()))
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(GRAPH_OUT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nAll episode data: {DETAIL_OUT.resolve()}")
print(f"Checkpoint summary: {SEED_OUT.resolve()}")
print(f"Paired comparison: {PAIR_OUT.resolve()}")
print(f"Graph: {GRAPH_OUT.resolve()}")
print("\nThese are preliminary results from short training runs.")
