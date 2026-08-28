"""Bài 18: Moving average"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def moving_average(values, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')

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

    window_size = 10
    ma_rewards = moving_average(episode_rewards, window_size)

    episodes = range(1, 101)
    ma_episodes = range(window_size, 101)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, alpha=0.4, label="Reward ban đầu")
    plt.plot(ma_episodes, ma_rewards, color='red', linewidth=2, label=f"Moving Average (window={window_size})")
    plt.title("Reward ban đầu và Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()

    figures_dir = "/content/RL_22010846_LeThiThuy/Lab01/figures"
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "moving_average.png")
    plt.savefig(save_path)
    
    print(f"Đã lưu biểu đồ vào: {save_path}")

    plt.show()

if __name__ == "__main__":
    main()
