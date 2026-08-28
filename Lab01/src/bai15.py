"""Bài 15: Tính thống kê reward"""

import gymnasium as gym
import numpy as np

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

    rewards_arr = np.array(episode_rewards)
    mean_reward = np.mean(rewards_arr)
    min_reward = np.min(rewards_arr)
    max_reward = np.max(rewards_arr)
    std_reward = np.std(rewards_arr)

    print(f"Mean reward : {mean_reward:.2f}")
    print(f"Min reward  : {min_reward:.2f}")
    print(f"Max reward  : {max_reward:.2f}")
    print(f"Std reward  : {std_reward:.2f}")

if __name__ == "__main__":
    main()
