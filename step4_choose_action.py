import gymnasium as gym
import numpy as np

# -------- 创建 6x6 FrozenLake --------
custom_map = [
    "SFFFFF",
    "FHFHFF",
    "FFFHFF",
    "HFFFFH",
    "FFHFHF",
    "FFFFFG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n

# -------- 初始化 Q-table --------
Q = np.zeros((num_states, num_actions))

# -------- 参数 ----------
epsilon = 0.2   # 20% 概率乱走

# -------- 随机选一个起始状态 ----------
state, info = env.reset()
print("Current state:", state)

# -------- ε-greedy 动作选择 ----------
if np.random.rand() < epsilon:
    action = env.action_space.sample()
    print("Action chosen: RANDOM", action)
else:
    action = np.argmax(Q[state])
    print("Action chosen: GREEDY", action)

env.close()
