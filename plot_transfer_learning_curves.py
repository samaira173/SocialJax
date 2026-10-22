
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results")
OUT.mkdir(exist_ok=True)

weights = [0.6, 0.8]
seeds = range(42, 48)
metric = "original_rewards"

# Load the six training runs for each reward weight.
frames = []

for weight in weights:
    for seed in seeds:
        filename = f"transfer_coin_s{weight}_seed{seed}.csv"
        path = OUT / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing training CSV: {path}")

        df = pd.read_csv(path)
        df["weight"] = weight
        df["training_seed"] = seed
        frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Confirm each run has compatible step positions.
counts = data.groupby(["weight", "env_step"])["training_seed"].nunique()
if not (counts == 6).all():
    raise ValueError(
        "Some weight/step combinations do not contain all six seeds."
    )

summary = (
    data.groupby(["weight", "env_step"])[metric]
    .agg(["mean", "std"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))

for weight in weights:
    group = summary[summary["weight"] == weight].sort_values("env_step")

    x = group["env_step"]
    mean = group["mean"]
    std = group["std"]

    ax.plot(x, mean, marker="o", markersize=3,
            linewidth=1.8, label=f"s = {weight}")
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.18,
    )

ax.set_xlim(0, 8192)
ax.set_xlabel("Environment steps")
ax.set_ylabel("Logged original environmental reward")
ax.set_title("Training reward across environment steps")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

output = OUT / "transfer_learning_curves.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

summary.to_csv(OUT / "transfer_learning_curve_summary.csv", index=False)

print("Saved graph:", output.resolve())
print("Saved summary:", (OUT / "transfer_learning_curve_summary.csv").resolve())
print("Training runs:", data["training_seed"].nunique(), "seeds per weight")
print("Steps per run:", data.groupby(["weight", "training_seed"]).size().unique())
print("Metric:", metric)
