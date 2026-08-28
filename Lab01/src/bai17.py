"""Bài 17: Vẽ reward theo episode"""

import os
import gymnasium as gym
import matplotlib.pyplot as plt

def random_agent(env):
    env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main():
    env = gym.make("CartPole-v1")
    episode_rewards = [random_agent(env) for _ in range(100)]
    env.close()

    episodes = range(1, 101)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, label="Total Reward")
    plt.title("Reward theo Episode trong CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    figures_dir = "/content/RL_22010846_LeThiThuy/Lab01/figures"
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "reward_cartpole.png")
    plt.savefig(save_path)
    
    print(f"Đã lưu biểu đồ vào: {save_path}")

    plt.show()

if __name__ == "__main__":
    main()
