"""Bài 31: Policy dựa trên observation"""
import gymnasium as gym
import numpy as np

def angle_based_policy(observation):
    # observation trong CartPole: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    pole_angle = observation[2]
    
    # Nếu thanh gậy nghiêng sang phải (> 0) thì đẩy xe sang phải (action 1) để nâng gậy lên
    # Ngược lại nếu nghiêng sang trái (< 0) thì đẩy xe sang trái (action 0)
    if pole_angle > 0:
        return 1
    else:
        return 0

def random_policy(observation, env):
    return env.action_space.sample()

def evaluate_policy(policy_type, n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            if policy_type == "angle":
                action = angle_based_policy(obs)
            else:
                action = random_policy(obs, env)
                
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))

def main():
    mean_angle = evaluate_policy("angle", n_episodes=100)
    mean_random = evaluate_policy("random", n_episodes=100)
    
    print(f"Reward trung bình với Angle-based Policy : {mean_angle:.2f}")
    print(f"Reward trung bình với Random Policy     : {mean_random:.2f}")

if __name__ == "__main__":
    main()
