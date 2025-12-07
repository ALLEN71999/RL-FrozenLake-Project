# Q-learning on FrozenLake Environment

This project implements **tabular Q-learning** to solve the FrozenLake reinforcement learning task using Python and Gymnasium.

The goal is to train an agent to reach the goal state while avoiding holes by interacting with the environment.

---

## 1. Environment

- Environment: `FrozenLake-v1` (Gymnasium)
- State space: Discrete grid positions
- Action space: {Left, Down, Right, Up}
- Reward:
  - +1 for reaching the goal
  - 0 otherwise
- Episode ends when the agent reaches the goal or falls into a hole

Two environments are used:
- **4×4 non-slippery FrozenLake** (main experiment)
- **Custom 6×6 FrozenLake** (extended experiment)

---

## 2. Algorithm

- Algorithm: Tabular Q-learning
- Policy: ε-greedy
- Update rule:

  Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]

- Key hyperparameters:
  - Learning rate (α)
  - Discount factor (γ)
  - Exploration rate (ε) with decay

---

## 3. File Structure

```text
rl_project/
├─ main_frozenlake_4x4.py        # Main experiment (4×4, non-slippery)
├─ frozenlake_6x6_experiment.py  # Extended experiment (6×6 map)
├─ q_table_6x6.npy               # Saved Q-table (6×6)
├─ reward_curve_6x6.png          # Training reward curve
├─ success_curve_6x6.png         # Training success rate curve
├─ step_by_step/                 # Step-by-step development scripts
│   ├─ step4_qtable_6x6.py
│   ├─ step4_choose_action.py
│   ├─ step4_update_once.py
│   └─ step4_episode_loop.py
└─ README.md
