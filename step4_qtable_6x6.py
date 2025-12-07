import gymnasium as gym
import numpy as np

# 自定义 6x6 地图
custom_map = [
    "SFFFFF",
    "FHFHFF",
    "FFFHFF",
    "HFFFFH",
    "FFHFHF",
    "FFFFFG"
]

# 创建 FrozenLake 环境（自定义地图）
env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=True
)

num_states = env.observation_space.n   # 应该是 36
num_actions = env.action_space.n       # 4

print("Number of states:", num_states)
print("Number of actions:", num_actions)

# 创建 Q-table
Q = np.zeros((num_states, num_actions))

print("\nQ-table shape:", Q.shape)
print("\nInitial Q-table:")
print(Q)

env.close()
