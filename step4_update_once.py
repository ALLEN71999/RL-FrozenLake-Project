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

# -------- 超参数（先固定） ----------
alpha = 0.8     # 学习率
gamma = 0.95    # 折扣因子
epsilon = 0.2   # 探索概率

# -------- reset 环境 ----------
state, info = env.reset()
print("Start state:", state)

# -------- 选动作（ε-greedy） ----------
if np.random.rand() < epsilon:
    action = env.action_space.sample()
    print("Action: RANDOM", action)
else:
    action = np.argmax(Q[state])
    print("Action: GREEDY", action)

# -------- 执行动作 ----------
next_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

print("Next state:", next_state)
print("Reward:", reward)
print("Done:", done)

# -------- Q-learning 更新（只更新一次） ----------
old_value = Q[state, action]
best_next = np.max(Q[next_state])

Q[state, action] = old_value + alpha * (reward + gamma * best_next - old_value)

print("\nQ value BEFORE update:", old_value)
print("Q value AFTER  update:", Q[state, action])

env.close()
