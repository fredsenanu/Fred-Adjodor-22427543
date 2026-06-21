"""
evaluate.py

Evaluates a trained Q-Learning agent (greedy policy, no exploration) over
N episodes and reports:
    - Success Rate (%)
    - Average Reward
    - Number of Failures
    - Number of Successful Runs
"""

import argparse
import json
import os
import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent


def evaluate(env, agent, n_episodes=100, max_steps=200):
    rewards = []
    successes = 0
    failures = 0
    lengths = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state, greedy=True)  # no exploration
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        if env.is_goal(state):
            successes += 1
        else:
            failures += 1

    results = {
        "n_episodes": n_episodes,
        "success_rate_pct": 100.0 * successes / n_episodes,
        "average_reward": float(np.mean(rewards)),
        "num_failures": failures,
        "num_successes": successes,
        "average_episode_length": float(np.mean(lengths)),
        "rewards": rewards,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent")
    parser.add_argument("--qtable", type=str, default="results/qtable_default.npy")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--slip_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    env = FrozenLakeEnv(
        is_slippery=args.slippery,
        slip_prob=args.slip_prob,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, seed=args.seed)
    agent.Q = np.load(args.qtable)
    agent.epsilon = 0.0  # purely greedy evaluation

    results = evaluate(env, agent, n_episodes=args.episodes, max_steps=args.max_steps)

    print(f"Evaluation over {args.episodes} episodes (Q-table: {args.qtable})")
    print(f"  Success Rate       : {results['success_rate_pct']:.2f}%")
    print(f"  Average Reward     : {results['average_reward']:.4f}")
    print(f"  Number of Successes: {results['num_successes']}")
    print(f"  Number of Failures : {results['num_failures']}")
    print(f"  Avg Episode Length : {results['average_episode_length']:.2f} steps")

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"eval_results_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results -> {out_path}")


if __name__ == "__main__":
    main()
