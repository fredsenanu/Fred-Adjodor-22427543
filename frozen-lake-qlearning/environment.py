"""
environment.py

A custom, from-scratch implementation of the Frozen Lake grid-world
environment. No Gymnasium / OpenAI Gym / any RL framework is used.

Grid legend:
    S : Start state
    F : Frozen (safe) state
    H : Hole (terminal, failure)
    G : Goal (terminal, success)

Actions:
    0 = Left
    1 = Down
    2 = Right
    3 = Up
"""

import random


class FrozenLakeEnv:
    """A simple, self-contained Frozen Lake environment.

    State representation: single integer index, computed as
        state = row * n_cols + col
    This keeps the Q-table a simple 2D array of shape (n_states, n_actions).
    """

    # Default 8x8 map exactly as specified in the assignment.
    DEFAULT_MAP = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

    ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    def __init__(self, desc=None, is_slippery=False, slip_prob=0.0,
                 step_reward=0.0, hole_reward=0.0, goal_reward=1.0,
                 max_steps=200, seed=None):
        """
        Args:
            desc: list[str] grid rows. Defaults to the 8x8 map above.
            is_slippery: if True, actions can randomly slip into a
                perpendicular direction (Bonus Option A: stochastic transitions).
            slip_prob: probability mass given to "unintended" directions when
                is_slippery=True. The intended action keeps (1 - slip_prob)
                probability; the remaining probability is split evenly between
                the two perpendicular directions (classic Frozen-Lake style).
            step_reward: reward for a normal (non-terminal) step.
            hole_reward: reward received when falling into a hole.
            goal_reward: reward received when reaching the goal.
            max_steps: episode horizon (truncation safeguard).
            seed: optional RNG seed for reproducibility.
        """
        self.desc = desc if desc is not None else list(self.DEFAULT_MAP)
        self.n_rows = len(self.desc)
        self.n_cols = len(self.desc[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self._rng = random.Random(seed)

        # Locate start, holes, goal from the map.
        self.start_state = None
        self.hole_states = set()
        self.goal_state = None
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                idx = self._to_index(r, c)
                cell = self.desc[r][c]
                if cell == "S":
                    self.start_state = idx
                elif cell == "H":
                    self.hole_states.add(idx)
                elif cell == "G":
                    self.goal_state = idx

        self.current_state = self.start_state
        self.n_steps = 0

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _to_index(self, row, col):
        return row * self.n_cols + col

    def _to_coords(self, state):
        return divmod(state, self.n_cols)

    # ------------------------------------------------------------------
    # Core API required by the assignment
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the agent to the start state. Returns the initial state."""
        self.current_state = self.start_state
        self.n_steps = 0
        return self.current_state

    def get_state(self):
        """Return the agent's current state."""
        return self.current_state

    def is_terminal(self, state=None):
        """Return True if `state` (or current state if None) is Hole or Goal."""
        s = self.current_state if state is None else state
        return s in self.hole_states or s == self.goal_state

    def _intended_delta(self, action):
        # 0=Left, 1=Down, 2=Right, 3=Up
        return {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }[action]

    def _perpendicular_actions(self, action):
        """Return the two actions perpendicular to `action` (for slipping)."""
        if action in (0, 2):     # Left/Right -> perpendicular is Up/Down
            return [1, 3]
        else:                    # Down/Up -> perpendicular is Left/Right
            return [0, 2]

    def _resolve_action(self, action):
        """Apply slip stochasticity (if enabled) and return the actually
        executed action."""
        if not self.is_slippery or self.slip_prob <= 0.0:
            return action

        # With probability (1 - slip_prob): intended action.
        # With probability slip_prob: split evenly across the two
        # perpendicular directions.
        r = self._rng.random()
        if r < (1.0 - self.slip_prob):
            return action
        else:
            perp = self._perpendicular_actions(action)
            return perp[0] if r < (1.0 - self.slip_prob / 2.0) else perp[1]

    def step(self, action):
        """Execute one timestep.

        Returns:
            next_state (int), reward (float), done (bool), info (dict)
        """
        if self.is_terminal():
            # Already terminal; no further movement, return as-is.
            return self.current_state, 0.0, True, {"reason": "already_terminal"}

        executed_action = self._resolve_action(action)
        row, col = self._to_coords(self.current_state)
        d_row, d_col = self._intended_delta(executed_action)

        new_row = row + d_row
        new_col = col + d_col

        # Enforce movement boundaries: if the move would leave the grid,
        # the agent stays in place.
        if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
            next_state = self._to_index(new_row, new_col)
        else:
            next_state = self.current_state  # bumped into a wall

        self.current_state = next_state
        self.n_steps += 1

        done = False
        reward = self.step_reward

        if next_state in self.hole_states:
            reward = self.hole_reward
            done = True
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        elif self.n_steps >= self.max_steps:
            done = True  # truncation, not a "success" or "failure" per se

        info = {
            "intended_action": action,
            "executed_action": executed_action,
            "slipped": executed_action != action,
            "steps": self.n_steps,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        """Print a human-readable view of the grid with the agent's position."""
        row, col = self._to_coords(self.current_state)
        lines = []
        for r in range(self.n_rows):
            line = []
            for c in range(self.n_cols):
                if r == row and c == col:
                    line.append("A")  # Agent
                else:
                    line.append(self.desc[r][c])
            lines.append(" ".join(line))
        print("\n".join(lines))
        print()

    # ------------------------------------------------------------------
    # Misc helpers used elsewhere in the project
    # ------------------------------------------------------------------
    def state_to_rowcol(self, state):
        return self._to_coords(state)

    def is_hole(self, state):
        return state in self.hole_states

    def is_goal(self, state):
        return state == self.goal_state


if __name__ == "__main__":
    # Quick smoke test
    env = FrozenLakeEnv()
    s = env.reset()
    print("Initial state:", s)
    env.render()
    s, r, done, info = env.step(2)  # move Right
    print("After moving Right ->", s, r, done, info)
    env.render()
