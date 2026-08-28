"""Bài 30: Policy luôn chọn một action"""
import gymnasium as gym
import numpy as np

def always_left_policy(observation):
    return 0  # Luôn di chuyển xe sang trái

def always_right_policy(observation):
    return 1  # Luôn di chuyển xe sang phải

def evaluate_policy(policy_fn, n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))

def main():
    mean_left = evaluate_policy(always_left_policy, n_episodes=100)
    mean_right = evaluate_policy(always_right_policy, n_episodes=100)
    
    print(f"Reward trung bình với Always Left Policy  : {mean_left:.2f}")
    print(f"Reward trung bình với Always Right Policy : {mean_right:.2f}")

if __name__ == "__main__":
    main()
