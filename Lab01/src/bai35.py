"""Bài 35: So sánh ba agent"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, policy, seed=None, max_steps=1000):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
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
    return {"reward": total_reward, "length": length}

def evaluate_policy_detailed(env_name, policy, n_episodes=500, seed=42):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        res = run_episode(env, policy, seed=ep_seed)
        rewards.append(res["reward"])
        lengths.append(res["length"])
    env.close()
    return rewards, {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths))
    }

# Các Policy
def random_policy(obs, env):
    return env.action_space.sample()

def angle_based_policy(obs):
    pole_angle = obs[2]
    return 1 if pole_angle > 0 else 0

def improved_policy(obs):
    pole_angle = obs[2]
    pole_angular_velocity = obs[3]
    return 1 if (pole_angle + 0.1 * pole_angular_velocity) > 0 else 0

def main():
    n_episodes = 500
    agents = {
        "Random": random_policy,
        "Angle-based": angle_based_policy,
        "Improved": improved_policy
    }
    
    results = {}
    history_rewards = {}
    
    print(f"{'Agent':<15} | {'Mean reward':<12} | {'Std':<10} | {'Min':<8} | {'Max':<8} | {'Mean length':<12}")
    print("-" * 75)
    
    for name, policy_fn in agents.items():
        rewards, stats = evaluate_policy_detailed("CartPole-v1", policy_fn, n_episodes=n_episodes, seed=42)
        results[name] = stats
        history_rewards[name] = rewards
        print(f"{name:<15} | {stats['mean_reward']:<12.2f} | {stats['std_reward']:<10.2f} | {stats['min_reward']:<8.1f} | {stats['max_reward']:<8.1f} | {stats['mean_length']:<12.2f}")

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 6))
    for name in agents:
        plt.plot(history_rewards[name], alpha=0.3, label=f"{name} (raw)")
        # Moving average 20 episodes
        mv_avg = np.convolve(history_rewards[name], np.ones(20)/20, mode='valid')
        plt.plot(range(19, n_episodes), mv_avg, linewidth=2, label=f"{name} (MA 20)")
        
    plt.title("So sánh Reward trung bình giữa các Agent (CartPole-v1)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_fig_path = "/content/RL_22010846_LeThiThuy/Lab01/figures/comparison_agents.png"
    plt.savefig(save_fig_path)
    plt.close()
    print(f"\nĐã lưu biểu đồ vào: {save_fig_path}")

    # Nhận xét:
    print("\n NHẬN XÉT ")
    print("1. Random policy cho hiệu năng thấp nhất với mean reward rất nhỏ và không thể giữ cân bằng gậy lâu.")
    print("2. Angle-based policy cải thiện rõ rệt so với Random policy nhờ phản ứng kịp thời theo hướng nghiêng của gậy.")
    print("3. Improved policy mang lại hiệu năng cao nhất nhờ kết hợp cả vận tốc góc, giúp dự đoán xu hướng di chuyển sớm hơn.")
    print("4. Biểu đồ đường trung bình trượt (Moving Average) thể hiện sự ổn định vượt trội của Improved policy so với 2 phương pháp còn lại.")
    print("5. Kết quả cho thấy việc tận dụng tối đa các thành phần thông tin trong Observation giúp tối ưu hóa hành vi điều khiển của Agent.")

if __name__ == "__main__":
    main()
