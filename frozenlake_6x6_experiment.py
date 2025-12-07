import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 6x6 自定义地图 ----------------
custom_map = [
    "SFFFFF",
    "FHFHFF",
    "FFFHFF",
    "HFFFFH",
    "FFHFHF",
    "FFFFFG"
]


def make_env(is_slippery: bool, render_mode=None):
    """创建 6x6 FrozenLake 环境，保证训练和评估一致"""
    return gym.make(
        "FrozenLake-v1",
        desc=custom_map,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )


# ---------------- 训练函数 ----------------
def train_q_learning(
    num_episodes: int = 5000,
    max_steps_per_episode: int = 200,
    alpha: float = 0.8,          # 学习率
    gamma: float = 0.99,         # 折扣因子
    epsilon_start: float = 1.0,  # 初始探索率
    epsilon_min: float = 0.05,   # 最小探索率
    epsilon_decay: float = 0.997 # 每个 episode 后的衰减
):
    # 训练时：不滑的环境（确定性，方便学习）
    env = make_env(is_slippery=False)

    n_states = env.observation_space.n   # 36
    n_actions = env.action_space.n       # 4

    # Q-table: (状态数 × 动作数)
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []   # 每局总 reward
    success_flags = []     # 是否到达终点（1 成功，0 失败）

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        last_reward = 0.0   # 记录本局最后一步的 reward

        for step in range(max_steps_per_episode):
            # ε-greedy 选动作
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # 探索
            else:
                action = int(np.argmax(Q[state]))    # 利用

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning 更新
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            last_reward = reward
            state = next_state

            if done:
                break

        # 一局结束后的记录
        episode_rewards.append(total_reward)
        success_flags.append(1 if last_reward > 0 else 0)

        # 衰减探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 打印训练进度
        if (episode + 1) % 500 == 0:
            last_100 = success_flags[-100:] if len(success_flags) >= 100 else success_flags
            avg_success = float(np.mean(last_100)) if last_100 else 0.0
            print(
                f"Episode {episode+1}/{num_episodes}, "
                f"epsilon={epsilon:.3f}, "
                f"recent success rate={avg_success:.3f}"
            )

    env.close()
    return Q, np.array(episode_rewards), np.array(success_flags)


# ---------------- 评估函数 ----------------
def evaluate_policy(Q: np.ndarray, num_episodes: int = 300, max_steps_per_episode: int = 200) -> float:
    # 评估时：一定要和训练环境完全一致！
    env = make_env(is_slippery=False)

    n_success = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        for _ in range(max_steps_per_episode):
            action = int(np.argmax(Q[state]))  # 只用 greedy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

            if done:
                if reward > 0:
                    n_success += 1
                break

    env.close()
    success_rate = n_success / num_episodes
    return success_rate


# ---------------- 画图辅助函数 ----------------
def moving_average(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_curves(rewards: np.ndarray, success_flags: np.ndarray):
    # 奖励曲线
    plt.figure()
    plt.plot(moving_average(rewards, window=50))
    plt.xlabel("Episode")
    plt.ylabel("Average reward (moving avg)")
    plt.title("Training reward on 6x6 FrozenLake")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve_6x6.png")
    print("Saved reward curve to reward_curve_6x6.png")

    # 成功率曲线
    if len(success_flags) > 0:
        plt.figure()
        plt.plot(moving_average(success_flags, window=50))
        plt.xlabel("Episode")
        plt.ylabel("Success rate (moving avg)")
        plt.title("Training success rate on 6x6 FrozenLake")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("success_curve_6x6.png")
        print("Saved success curve to success_curve_6x6.png")


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    print("Start training Q-learning on 6x6 FrozenLake (non-slippery)...\n")

    Q, episode_rewards, success_flags = train_q_learning()

    print("\nTraining finished. Evaluating learned policy...")
    success_rate = evaluate_policy(Q)
    print(f"Evaluation success rate: {success_rate:.3f}")

    # 保存 Q-table
    np.save("q_table_6x6.npy", Q)
    print("Saved Q-table to q_table_6x6.npy")

    # 画学习曲线
    plot_curves(episode_rewards, success_flags)
