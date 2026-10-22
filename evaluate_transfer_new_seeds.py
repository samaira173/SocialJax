
from pathlib import Path
import csv

import jax
import numpy as np
import socialjax

from algorithms.utils.io_utils import load_params
from algorithms.utils.networks import ActorCritic


CHECKPOINTS = {
    "transfer_s0.6_seed42": Path("checkpoints/transfer_s0_6_seed42.pkl"),
    "transfer_s0.8_seed42": Path("checkpoints/transfer_s0_8_seed42.pkl"),
    "transfer_s0.6_seed43": Path("checkpoints/transfer_s0_6_seed43.pkl"),
    "transfer_s0.8_seed43": Path("checkpoints/transfer_s0_8_seed43.pkl"),
    "transfer_s0.6_seed44": Path("checkpoints/transfer_s0_6_seed44.pkl"),
    "transfer_s0.8_seed44": Path("checkpoints/transfer_s0_8_seed44.pkl"),
    "transfer_s0.6_seed45": Path("checkpoints/transfer_s0_6_seed45.pkl"),
    "transfer_s0.8_seed45": Path("checkpoints/transfer_s0_8_seed45.pkl"),
    "transfer_s0.6_seed46": Path("checkpoints/transfer_s0_6_seed46.pkl"),
    "transfer_s0.8_seed46": Path("checkpoints/transfer_s0_8_seed46.pkl"),
    "transfer_s0.6_seed47": Path("checkpoints/transfer_s0_6_seed47.pkl"),
    "transfer_s0.8_seed47": Path("checkpoints/transfer_s0_8_seed47.pkl"),
}

OUTPUT = Path("results/transfer_evaluation_new_seeds.csv")
NUM_EPISODES = 10
MAX_STEPS = 1000


def evaluate_checkpoint(condition, checkpoint_path, env, network):
    params = load_params(str(checkpoint_path))
    rows = []

    for episode in range(NUM_EPISODES):
        episode_seed = 1000 + episode
        key = jax.random.PRNGKey(episode_seed)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        agent_returns = np.zeros(2, dtype=np.float64)
        own_coin_metric_sums = np.zeros(2, dtype=np.float64)

        for step in range(MAX_STEPS):
            key, action_key = jax.random.split(key)

            # Same parameter-sharing network for both agents.
            pi, _ = network.apply(params, obs)
            actions = np.asarray(
                pi.sample(seed=action_key)
            ).reshape(-1).astype(int).tolist()

            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(
                step_key, state, actions
            )

            agent_returns += np.asarray(
                rewards, dtype=np.float64
            ).reshape(2)

            own_coin_metric_sums += np.asarray(
                info["eat_own_coins"], dtype=np.float64
            ).reshape(2)

            if bool(np.asarray(dones["__all__"])):
                break

        rows.append({
            "trained_policy": condition,
            "checkpoint": checkpoint_path.name,
            "episode": episode + 1,
            "episode_seed": episode_seed,
            "steps": step + 1,
            "agent0_environment_return": agent_returns[0],
            "agent1_environment_return": agent_returns[1],
            "sum_agent_environment_returns": agent_returns.sum(),
            "agent0_eat_own_coins_metric_sum": own_coin_metric_sums[0],
            "agent1_eat_own_coins_metric_sum": own_coin_metric_sums[1],
        })

        print(
            f"{condition}: episode {episode + 1}/{NUM_EPISODES}; "
            f"agent returns={agent_returns.tolist()}; "
            f"sum={agent_returns.sum():.2f}"
        )

    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Choose another output filename "
            "before running again. Existing results were not overwritten."
        )

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    # Both policies are evaluated under the SAME reward configuration.
    env = socialjax.make(
    	"coin_game",
    	num_agents=2,
    	num_inner_steps=MAX_STEPS,
    	cnn=True,
    	jit=True,
    	shared_rewards=False,
    	interest=False,
	)

    network = ActorCritic(
        action_dim=env.action_space().n,
        activation="relu",
    )

    all_rows = []
    for condition, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating {condition}")
        all_rows.extend(
            evaluate_checkpoint(condition, checkpoint_path, env, network)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nEvaluation complete.")
    print("Episodes recorded:", len(all_rows))
    print("Saved to:", OUTPUT.resolve())
    print(
        "Both policies were measured in the same individual-reward "
        "environment. These are still preliminary evaluations."
    )


if __name__ == "__main__":
    main()
