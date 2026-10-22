
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/transfer_evaluation_new_seeds.csv")
OUTPUT = Path("results/transfer_agent_return_scatter.png")

df = pd.read_csv(INPUT)

# Extract the reward weight from each policy label.
df["s_interest"] = df["trained_policy"].str.extract(
    r"s(0\.\d+)_seed"
).astype(float)

fig, ax = plt.subplots(figsize=(7, 6))

for weight in [0.6, 0.8]:
    subset = df[df["s_interest"] == weight]

    ax.scatter(
        subset["agent0_environment_return"],
        subset["agent1_environment_return"],
        alpha=0.65,
        s=45,
        label=f"s = {weight}",
    )

# Equal-return reference line.
values = pd.concat([
    df["agent0_environment_return"],
    df["agent1_environment_return"],
])
low, high = values.min(), values.max()
padding = (high - low) * 0.05 if high > low else 1

low -= padding
high += padding

ax.plot(
    [low, high], [low, high],
    linestyle="--",
    color="gray",
    label="Equal returns (y = x)",
)

ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Agent 0 environmental return")
ax.set_ylabel("Agent 1 environmental return")
ax.set_title("Agent outcomes by reward-mixing weight")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT.resolve())
print("Episodes plotted:", len(df))
print("\nMean agent returns by weight:")
print(
    df.groupby("s_interest")[
        ["agent0_environment_return", "agent1_environment_return"]
    ].mean().round(2)
)
