"""
visualize.py

Generates training-performance graphs from saved results/train_stats_*.json
files (Bonus Option B: Visualize training performance using graphs).

Produces:
    1. results/plot_learning_curve_default.png
       - Episode reward (smoothed) and rolling success rate vs episode, with
         epsilon overlay, for the default hyperparameter run.
    2. results/plot_hyperparam_comparison.png
       - Rolling success-rate curves comparing different alpha/gamma settings.
    3. results/plot_exploration_comparison.png
       - Pure epsilon-greedy vs decaying epsilon-greedy (Bonus Option C).
    4. results/plot_stochastic_comparison.png
       - Deterministic vs stochastic (slippery) training curves.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def load_stats(tag):
    path = os.path.join(RESULTS_DIR, f"train_stats_{tag}.json")
    with open(path) as f:
        return json.load(f)


def rolling_mean(x, window=100):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_learning_curve(tag="default", window=100):
    stats = load_stats(tag)
    rewards = stats["episode_rewards"]
    success = stats["episode_success"]
    epsilon = stats["epsilon_history"]

    smoothed_reward = rolling_mean(rewards, window)
    smoothed_success = rolling_mean(success, window) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(smoothed_reward, color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel(f"Avg Reward (rolling {window}-ep)")
    ax1.set_title("Q-Learning Training Performance — Frozen Lake 8x8")
    ax1.grid(alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(epsilon, color="#d62728", linewidth=1, alpha=0.6, linestyle="--")
    ax1b.set_ylabel("Epsilon", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")

    ax2.plot(smoothed_success, color="#2ca02c", linewidth=1.5)
    ax2.set_ylabel(f"Success Rate % (rolling {window}-ep)")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"plot_learning_curve_{tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hyperparam_comparison(window=100):
    configs = [
        ("default", "alpha=0.1, gamma=0.99 (default)"),
        ("alpha_low", "alpha=0.02 (low)"),
        ("alpha_high", "alpha=0.5 (high)"),
        ("gamma_low", "gamma=0.80 (low)"),
        ("gamma_high", "gamma=0.999 (high)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, label in configs:
        try:
            stats = load_stats(tag)
        except FileNotFoundError:
            continue
        smoothed = rolling_mean(stats["episode_success"], window) * 100
        ax.plot(smoothed, label=label, linewidth=1.3)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success Rate % (rolling {window}-ep)")
    ax.set_title("Effect of Learning Rate (alpha) and Discount Factor (gamma)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "plot_hyperparam_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_exploration_comparison(window=100):
    """Bonus Option C: Pure epsilon-greedy vs Decaying epsilon-greedy."""
    configs = [
        ("default", "Decaying epsilon-greedy (1.0 -> 0.01)"),
        ("pure_epsilon_greedy", "Pure epsilon-greedy (fixed epsilon=0.1)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for tag, label in configs:
        try:
            stats = load_stats(tag)
        except FileNotFoundError:
            continue
        smoothed_success = rolling_mean(stats["episode_success"], window) * 100
        smoothed_reward = rolling_mean(stats["episode_rewards"], window)
        axes[0].plot(smoothed_success, label=label, linewidth=1.3)
        axes[1].plot(smoothed_reward, label=label, linewidth=1.3)

    axes[0].set_title("Success Rate")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Success Rate % (rolling {window}-ep)")
    axes[0].set_ylim(-5, 105)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_title("Average Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Avg Reward (rolling {window}-ep)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Pure epsilon-Greedy vs Decaying epsilon-Greedy")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "plot_exploration_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_stochastic_comparison(window=100):
    """Bonus Option A: Deterministic vs Stochastic transitions."""
    configs = [
        ("default", "Deterministic environment"),
        ("slippery", "Stochastic / slippery environment (p_slip=0.2)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, label in configs:
        try:
            stats = load_stats(tag)
        except FileNotFoundError:
            continue
        smoothed = rolling_mean(stats["episode_success"], window) * 100
        ax.plot(smoothed, label=label, linewidth=1.3)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success Rate % (rolling {window}-ep)")
    ax.set_title("Deterministic vs Stochastic Transitions: Training Convergence")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "plot_stochastic_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_learning_curve("default")
    plot_hyperparam_comparison()
    plot_exploration_comparison()
    plot_stochastic_comparison()
