# Frozen Lake from First Principles — Q-Learning

A complete, from-scratch implementation of the classic Frozen Lake
grid-world and a tabular Q-Learning agent that learns to solve it. No
Gymnasium, OpenAI Gym, Stable-Baselines, RLlib, or any other RL framework is
used anywhere in this project — the environment, the agent, the training
loop, and the evaluation harness are all plain Python / NumPy.

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning in which an
**agent** learns to make decisions by interacting with an **environment**.
At each timestep the agent observes a **state**, selects an **action**, and
receives a **reward** along with a new state. There is no labelled dataset
telling the agent what the "correct" action is — instead, the agent must
discover good behaviour purely through trial and error, guided by the
reward signal. The agent's goal is to learn a **policy** (a mapping from
states to actions) that maximizes its expected cumulative (discounted)
reward over time.

### What is Frozen Lake?

Frozen Lake is a classic, deceptively simple RL benchmark. The agent stands
on a frozen grid and must walk from a **Start** cell to a **Goal** cell
without falling into a **Hole**. Some cells are safe **Frozen** tiles. The
challenge is that:

- Falling in a hole ends the episode with no reward.
- Reaching the goal ends the episode with a reward of +1.
- The agent initially has no idea which moves are safe, so it must explore
  the grid, repeatedly fail, and gradually learn a policy that reliably
  reaches the goal.

This project uses the standard 8×8 map:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

## Environment Design

Implemented in `environment.py` as the `FrozenLakeEnv` class.

### State representation

States are represented as a **single integer index**:

```
state = row * n_cols + col
```

For the 8×8 map this gives 64 states (0–63). This keeps the Q-table a
simple 2-D NumPy array of shape `(64, 4)`, indexable directly by state and
action — no need for a dictionary or coordinate-to-index lookup table
outside the environment.

### Action representation

Four discrete actions, matching the assignment specification exactly:

| Action | Meaning | (Δrow, Δcol) |
|--------|---------|--------------|
| 0      | Left    | (0, -1)      |
| 1      | Down    | (+1, 0)      |
| 2      | Right   | (0, +1)      |
| 3      | Up      | (-1, 0)      |

Moves that would take the agent off the edge of the grid are **boundary
clamped** — the agent simply stays in its current cell ("bumps into the
wall") rather than wrapping or being penalized.

### Reward structure

| Event                  | Reward |
|-------------------------|--------|
| Normal (non-terminal) step | 0.0 |
| Falling into a Hole (H)    | 0.0 (episode ends) |
| Reaching the Goal (G)      | +1.0 (episode ends) |

This is the standard sparse Frozen Lake reward: the only signal the agent
ever receives is at the very end of a successful episode, which is what
makes the credit-assignment problem genuinely interesting — the agent has
to propagate that single +1 backward through every state on the path that
led to it, which is exactly what the Q-Learning update rule does over many
episodes.

### Stochastic transitions (optional)

The environment optionally supports **slippery** dynamics
(`is_slippery=True`, `slip_prob`), in which the executed action can
randomly differ from the intended action — the agent's move "slips" into
one of the two perpendicular directions. This is used for the Bonus Task
(see below) and is disabled by default.

## Q-Learning Algorithm

Implemented in `agent.py` as the `QLearningAgent` class.

### Description of Q-Learning

Q-Learning is a **model-free, off-policy** temporal-difference RL
algorithm. It learns a table of action-values, `Q(s, a)`, estimating the
expected discounted future reward of taking action `a` in state `s` and
acting optimally afterward. Because it is off-policy, the agent can learn
the optimal Q-values while following an exploratory (non-optimal) policy
during training — which is essential, since an agent that only ever
exploits its current (initially-random) knowledge would never discover the
path to the goal.

### The update equation

After every transition `(s, a, r, s')`, the Q-table is updated using
exactly the rule specified in the assignment:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

- **alpha (learning rate)** controls how much new information overrides old
  estimates.
- **gamma (discount factor)** controls how much future reward matters
  relative to immediate reward.
- **max_a' Q(s', a')** is the agent's current best estimate of the value of
  the next state — this is the "bootstrapping" term that lets value
  propagate backward through the state space over many episodes.
- The bracketed term `r + gamma * max_a' Q(s', a') - Q(s, a)` is the
  **TD-error**: the difference between what we just observed and what we
  previously believed.

When `s'` is terminal, the bootstrapped term is treated as 0, since there
is no future reward from a terminal state.

### Exploration strategy

The agent uses **epsilon-greedy** exploration:

- With probability `epsilon`, take a uniformly random action (explore).
- Otherwise, take the action with the highest current Q-value (exploit),
  breaking ties uniformly at random.

`epsilon` starts high (`1.0`, fully random) and **decays multiplicatively**
after every episode (`epsilon <- max(epsilon_min, epsilon * epsilon_decay)`)
down to a small floor (`0.01`), so the agent explores heavily early on and
gradually shifts toward exploiting what it has learned.

## Training Procedure

### Hyperparameters used (default configuration)

| Hyperparameter      | Value   |
|---------------------|---------|
| Episodes            | 20,000  |
| Learning rate (α)   | 0.1     |
| Discount factor (γ) | 0.99    |
| Epsilon start       | 1.0     |
| Epsilon min         | 0.01    |
| Epsilon decay       | 0.9995  |
| Max steps / episode | 200     |

Several alternative configurations were also trained and compared (see
`results/` and the technical report) by varying α, γ, and the epsilon decay
rate, as required by the assignment.

### Number of episodes

20,000 episodes per configuration (40,000 for the stochastic/slippery
bonus run, since stochastic dynamics require more experience to converge).

## Results

### Final success rate

The default configuration converges to a **100% rolling success rate**
within roughly the first 5,000–9,000 episodes and is evaluated at:

- **Success rate:** 100.00% (over 200 greedy evaluation episodes)
- **Average reward:** 1.0000
- **Average episode length:** 14.0 steps (the shortest safe path length)

### Learned policy

```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → → ↓ ↓ ↓
↑ ↑ ↑ H → → → ↓
↑ ↑ ↑ H ↑ → → ↓
↑ ↑ ↑ H → → → ↓
↑ H H → ↑ ↑ H ↓
← H ← ← H ↑ H ↓
← ← ← H ← ← ← G
```

(`H` = hole, `G` = goal; arrows show the greedy action learned for every
other state.)

### Discussion of performance

All tested hyperparameter combinations (α ∈ {0.02, 0.1, 0.5}, γ ∈ {0.80,
0.99, 0.999}) eventually converge to a 100% success rate on this
deterministic map, because the state space is small (64 states) and the
reward signal, while sparse, is unambiguous. The settings mainly affect
**convergence speed** and the **smoothness** of the learning curve rather
than the final outcome — see the technical report and the plots in
`results/` for episode-by-episode comparisons. Performance degrades
gracefully (rather than catastrophically) once stochastic transitions are
introduced, and an agent trained directly on the stochastic dynamics
learns a visibly more cautious policy than one trained deterministically
and merely tested under stochasticity.

## Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent (default hyperparameters)
python train.py --episodes 20000 --alpha 0.1 --gamma 0.99 --tag default

# 3. Evaluate the trained agent over 100+ episodes
python evaluate.py --episodes 200 --qtable results/qtable_default.npy --tag default

# 4. Generate training-performance plots (Bonus Option B)
python visualize.py
```

### Reproducing the hyperparameter experiments

```bash
python train.py --episodes 20000 --alpha 0.02 --gamma 0.99 --tag alpha_low
python train.py --episodes 20000 --alpha 0.5  --gamma 0.99 --tag alpha_high
python train.py --episodes 20000 --alpha 0.1  --gamma 0.80 --tag gamma_low
python train.py --episodes 20000 --alpha 0.1  --gamma 0.999 --tag gamma_high
python train.py --episodes 20000 --alpha 0.1  --gamma 0.99 --epsilon_decay 0.999  --tag eps_fast
python train.py --episodes 20000 --alpha 0.1  --gamma 0.99 --epsilon_decay 0.9999 --tag eps_slow
```

### Reproducing the bonus tasks

**Option A — Stochastic transitions:**
```bash
python train.py --episodes 40000 --slippery --slip_prob 0.2 --tag slippery
python evaluate.py --episodes 500 --qtable results/qtable_slippery.npy --slippery --slip_prob 0.2 --tag slippery
```

**Option C — Pure vs. decaying epsilon-greedy:**
```bash
python train.py --episodes 20000 --epsilon_start 0.1 --epsilon_min 0.1 --epsilon_decay 1.0 --tag pure_epsilon_greedy
```

All scripts accept `--help` for the full list of options.

## Project Structure

```
frozen-lake-qlearning/
├── environment.py     # FrozenLakeEnv: reset, step, render, get_state, is_terminal
├── agent.py           # QLearningAgent: epsilon-greedy action selection, Q-update
├── train.py           # Training loop, statistics logging, policy extraction
├── evaluate.py        # Greedy-policy evaluation harness
├── visualize.py        # Training-performance graphs (Bonus Option B)
├── requirements.txt
├── README.md
├── report.pdf
└── results/            # Saved Q-tables, training stats, evaluation results, plots
```
