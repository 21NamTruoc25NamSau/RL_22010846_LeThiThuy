"""Bài 20: So sánh hai seed"""
import gymnasium as gym
import numpy as np

def run_episodes(seed, n_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        
    env.close()
    return np.mean(rewards)

def main():
    mean_42 = run_episodes(seed=42, n_episodes=20)
    mean_100 = run_episodes(seed=100, n_episodes=20)
    
    print(f"Reward trung bình với seed = 42 : {mean_42:.2f}")
    print(f"Reward trung bình với seed = 100: {mean_100:.2f}")

if __name__ == "__main__":
    main()
