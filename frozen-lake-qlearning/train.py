"""
train.py

Trains a Q-Learning agent on the custom FrozenLakeEnv from first
principles. Records training statistics (episode rewards, success rate,
epsilon decay), extracts the learned policy, and saves results/artifacts
to the results/ directory.

Usage:
    python train.py
    python train.py --episodes 20000 --alpha 0.1 --gamma 0.99
"""

import argparse
import json
import os
import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent

ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}  # L D R U


def train(env, agent, n_episodes=20000, max_steps=200, log_every=1000,
           window=100):
    """Run the Q-Learning training loop.

    Returns a dict of recorded training statistics.
    """
    episode_rewards = []
    episode_lengths = []
    episode_success = []  # 1 if goal reached, 0 otherwise
    epsilon_history = []

    successes_so_far = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        reached_goal = env.is_goal(state)
        if reached_goal:
            successes_so_far += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_success.append(1 if reached_goal else 0)
        epsilon_history.append(agent.epsilon)

        agent.decay_epsilon()

        if log_every and ep % log_every == 0:
            recent_success_rate = np.mean(episode_success[-window:]) * 100
            recent_avg_reward = np.mean(episode_rewards[-window:])
            print(f"Episode {ep:>6} | "
                  f"epsilon={agent.epsilon:.4f} | "
                  f"success_rate(last {window})={recent_success_rate:5.1f}% | "
                  f"avg_reward(last {window})={recent_avg_reward:.3f} | "
                  f"total_successes={successes_so_far}")

    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_success": episode_success,
        "epsilon_history": epsilon_history,
        "total_successes": successes_so_far,
        "n_episodes": n_episodes,
    }
    return stats


def extract_policy_grid(env, agent):
    """Build a human-readable grid of the greedy policy."""
    policy = agent.get_policy()
    grid_lines = []
    for r in range(env.n_rows):
        row_symbols = []
        for c in range(env.n_cols):
            s = env._to_index(r, c)
            cell = env.desc[r][c]
            if cell == "H":
                row_symbols.append("H")
            elif cell == "G":
                row_symbols.append("G")
            else:
                row_symbols.append(ACTION_ARROWS[int(policy[s])])
        grid_lines.append(" ".join(row_symbols))
    return "\n".join(grid_lines), policy


def main():
    parser = argparse.ArgumentParser(description="Train Q-Learning agent on FrozenLakeEnv")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--slippery", action="store_true",
                         help="Enable stochastic transitions (Bonus Option A)")
    parser.add_argument("--slip_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="default",
                         help="Name used for saved result files (for hyperparameter comparisons)")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    env = FrozenLakeEnv(
        is_slippery=args.slippery,
        slip_prob=args.slip_prob,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print(f"Training Q-Learning agent for {args.episodes} episodes "
          f"(alpha={args.alpha}, gamma={args.gamma}, "
          f"epsilon_decay={args.epsilon_decay}, slippery={args.slippery})\n")

    stats = train(env, agent, n_episodes=args.episodes, max_steps=args.max_steps)

    overall_success_rate = 100.0 * stats["total_successes"] / args.episodes
    print(f"\nTraining complete.")
    print(f"Total successes: {stats['total_successes']} / {args.episodes} "
          f"({overall_success_rate:.2f}%)")

    policy_grid_str, policy_array = extract_policy_grid(env, agent)
    print("\nLearned policy (grid form):")
    print(policy_grid_str)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    qtable_path = os.path.join(args.results_dir, f"qtable_{args.tag}.npy")
    np.save(qtable_path, agent.Q)

    stats_path = os.path.join(args.results_dir, f"train_stats_{args.tag}.json")
    with open(stats_path, "w") as f:
        json.dump({
            "args": vars(args),
            "total_successes": stats["total_successes"],
            "overall_success_rate_pct": overall_success_rate,
            "episode_rewards": stats["episode_rewards"],
            "episode_success": stats["episode_success"],
            "epsilon_history": stats["epsilon_history"],
            "policy": policy_array.tolist(),
            "policy_grid": policy_grid_str,
        }, f)

    print(f"\nSaved Q-table -> {qtable_path}")
    print(f"Saved training stats -> {stats_path}")


if __name__ == "__main__":
    main()
