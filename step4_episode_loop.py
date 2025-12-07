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
alpha = 0.8
gamma = 0.95
epsilon = 0.2

# -------- 开始一个 episode ----------
state, info = env.reset()
done = False
step_count = 0

print("Episode start")

while not done:
    step_count += 1

    # ε-greedy 选动作
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
        action_type = "RANDOM"
    else:
        action = np.argmax(Q[state])
        action_type = "GREEDY"

    # 执行动作
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Q-learning 更新
    old_value = Q[state, action]
    best_next = np.max(Q[next_state])
    Q[state, action] = old_value + alpha * (reward + gamma * best_next - old_value)

    print(
        f"Step {step_count}: "
        f"state={state}, action={action_type}({action}), "
        f"reward={reward}, next_state={next_state}"
    )

    state = next_state

print("\nEpisode finished")
env.close()
