"""Bài 22: Hàm thí nghiệm có seed"""
import gymnasium as gym
import numpy as np

def experiment(seed, n_episodes=20):
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    
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
    
    rewards_arr = np.array(rewards)
    return {
        "seed": seed,
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "min_reward": float(np.min(rewards_arr))
    }

def main():
    seeds = [10, 20, 30, 42, 100]
    for s in seeds:
        res = experiment(seed=s, n_episodes=20)
        print(f"Seed {res['seed']:<3} | Mean: {res['mean_reward']:<6.2f} | Std: {res['std_reward']:<6.2f} | Max: {res['max_reward']:<5.1f} | Min: {res['min_reward']:<5.1f}")

if __name__ == "__main__":
    main()
