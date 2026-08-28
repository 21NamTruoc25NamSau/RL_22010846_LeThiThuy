"""Bài 34: Xây dựng hàm evaluate_policy()"""
import gymnasium as gym
import numpy as np

def run_episode(env, policy, seed=None, max_steps=1000):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False
    
    for _ in range(max_steps):
        try:
            action = policy(obs, env)
        except TypeError:
            action = policy(obs)
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        
        if terminated or truncated:
            break
            
    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated
    }

def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    """
    Đánh giá policy trên môi trường cụ thể qua n_episodes.
    """
    env = gym.make(env_name)
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        res = run_episode(env, policy, seed=ep_seed)
        rewards.append(res["reward"])
        lengths.append(res["length"])
        
    env.close()
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths))
    }

def random_policy(observation, env):
    return env.action_space.sample()

def main():
    metrics = evaluate_policy("CartPole-v1", random_policy, n_episodes=100, seed=42)
    print("Kết quả đánh giá Random Policy trên CartPole-v1 (100 episodes):")
    for key, val in metrics.items():
        print(f"  {key:<12}: {val:.2f}")

if __name__ == "__main__":
    main()
