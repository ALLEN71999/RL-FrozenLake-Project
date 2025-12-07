import gymnasium as gym
import numpy as np

def train_and_eval():
    env = gym.make("FrozenLake-v1", is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    alpha = 0.8
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # -------- 训练 --------
    for episode in range(3000):
        state, _ = env.reset()
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # -------- 评估 --------
    success = 0
    for _ in range(200):
        state, _ = env.reset()
        done = False

        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward > 0:
                success += 1
                break

    env.close()
    print("Evaluation success rate:", success / 200)

if __name__ == "__main__":
    train_and_eval()
