"""Bài 28: So sánh deterministic và stochastic"""
import gymnasium as gym
import numpy as np

def evaluate_policy(is_slippery, n_episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    
    successes = 0
    rewards = []
    lengths = []
    
    for _ in range(n_episodes):
        env.reset()
        total_reward = 0.0
        length = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            if terminated or truncated:
                if reward == 1.0:
                    successes += 1
                break
        rewards.append(total_reward)
        lengths.append(length)
        
    env.close()
    
    return {
        "success_rate": successes / n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths))
    }

def main():
    det_results = evaluate_policy(is_slippery=False, n_episodes=500)
    stoch_results = evaluate_policy(is_slippery=True, n_episodes=500)
    
    print(f"{'Thông số':<25} | {'Deterministic (False)':<22} | {'Stochastic (True)':<22}")
    print("-" * 75)
    print(f"{'Success Rate':<25} | {det_results['success_rate']:<22.2%} | {stoch_results['success_rate']:<22.2%}")
    print(f"{'Average Reward':<25} | {det_results['avg_reward']:<22.4f} | {stoch_results['avg_reward']:<22.4f}")
    print(f"{'Average Episode Length':<25} | {det_results['avg_length']:<22.2f} | {stoch_results['avg_length']:<22.2f}")

    # KẾT LUẬN:
    # 1. Khi is_slippery=False (Deterministic), agent di chuyển chính xác theo action đã chọn, tuy nhiên hành động ngẫu nhiên vẫn có tỉ lệ lặp hoặc rơi xuống hố cao.
    # 2. Khi is_slippery=True (Stochastic/Ngẫu nhiên do trượt), hành động thực thi có xác suất bị lệch sang các hướng khác. Điều này khiến tỉ lệ thành công của Random Policy giảm rõ rệt và độ dài episode ngắn hơn do dễ trượt xuống hố sớm hơn.

if __name__ == "__main__":
    main()
