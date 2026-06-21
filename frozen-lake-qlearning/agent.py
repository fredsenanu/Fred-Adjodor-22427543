"""
agent.py

A Q-Learning agent implemented entirely from first principles (no RL
frameworks). Implements:
    - Q-table initialization
    - Epsilon-greedy exploration with decay
    - The exact Q-Learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""

import random
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 seed=None):
        """
        Args:
            n_states: number of discrete states in the environment.
            n_actions: number of discrete actions available.
            alpha: learning rate (step size).
            gamma: discount factor.
            epsilon_start: initial exploration rate.
            epsilon_min: floor value epsilon decays towards.
            epsilon_decay: multiplicative decay applied to epsilon after
                each episode (epsilon <- max(epsilon_min, epsilon * decay)).
            seed: optional seed for reproducibility.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table initialization: all zeros. Shape = (n_states, n_actions).
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection.

        Args:
            state: current state index.
            greedy: if True, ignore epsilon and always act greedily
                (used at evaluation time).
        """
        if not greedy and self._rng.random() < self.epsilon:
            # Explore: pick a uniformly random action.
            return self._rng.randrange(self.n_actions)
        # Exploit: pick the action with the highest Q-value.
        # Break ties randomly so the agent doesn't always default to
        # action 0 when several actions are equally (e.g. initially) good.
        row = self.Q[state]
        max_value = np.max(row)
        best_actions = np.flatnonzero(row == max_value)
        return self._rng.choice(best_actions.tolist())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """Apply the Q-Learning update rule:

            Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        When `done` is True, the next-state value is treated as 0 because
        terminal states have no future reward.
        """
        best_next_q = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        """Decay epsilon multiplicatively, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        self.epsilon = self.epsilon_start

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------
    def get_greedy_action(self, state):
        return int(np.argmax(self.Q[state]))

    def get_policy(self):
        """Return an array of length n_states with the greedy action
        for every state."""
        return np.argmax(self.Q, axis=1)


if __name__ == "__main__":
    # Quick smoke test
    agent = QLearningAgent(n_states=64, n_actions=4, seed=0)
    s, a, r, s2, done = 0, 2, 0.0, 1, False
    print("Q before:", agent.Q[0])
    agent.update(s, a, r, s2, done)
    print("Q after:", agent.Q[0])
    print("Epsilon before decay:", agent.epsilon)
    agent.decay_epsilon()
    print("Epsilon after decay:", agent.epsilon)
