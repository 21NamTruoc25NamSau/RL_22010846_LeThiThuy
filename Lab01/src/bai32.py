"""Bài 32: Cải tiến heuristic"""
import gymnasium as gym
import numpy as np

def improved_heuristic_policy(observation):
    # Lấy 2 thành phần: góc nghiêng và vận tốc góc của gậy
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    
    # Kết hợp góc và vận tốc góc để dự đoán xu hướng nghiêng của gậy
    # Tổng hợp lực/mô-men dự kiến:
    total_signal = pole_angle + 0.1 * pole_angular_velocity
    
    if total_signal > 0:
        return 1  # Đẩy sang phải
    else:
        return 0  # Đẩy sang trái

def random_policy(env):
    return env.action_space.sample()

def evaluate(policy_fn, n_episodes=100, is_random=False):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = random_policy(env) if is_random else policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))

def main():
    mean_improved = evaluate(improved_heuristic_policy, n_episodes=100)
    mean_random = evaluate(None, n_episodes=100, is_random=True)
    
    print(f"Mean reward của Improved Heuristic Policy : {mean_improved:.2f}")
    print(f"Mean reward của Random Policy             : {mean_random:.2f}")
    print(f"Kiểm tra tiêu chí (Improved > Random)    : {mean_improved > mean_random}")

if __name__ == "__main__":
    main()
