import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def create_environment(env_name="CartPole-v1"):
    """Khởi tạo môi trường theo Gymnasium."""
    return gym.make(env_name)

def policy(observation, env=None):
    """
    Policy cải tiến kết hợp pole_angle và pole_angular_velocity.
    """
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    return 1 if (pole_angle + 0.1 * pole_angular_velocity) > 0 else 0

def random_policy(observation, env):
    """Policy hành động ngẫu nhiên."""
    return env.action_space.sample()

def run_episode(env, policy_fn, seed=None, max_steps=1000):
    """Chạy 1 episode duy nhất và thu thập kết quả."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False
    
    for _ in range(max_steps):
        try:
            action = policy_fn(obs, env)
        except TypeError:
            action = policy_fn(obs)
            
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

def evaluate_policy(env_name, policy_fn, n_episodes=500, seed=42):
    """Đánh giá policy qua n_episodes và trả về dữ liệu thống kê."""
    env = create_environment(env_name)
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        res = run_episode(env, policy_fn, seed=ep_seed)
        rewards.append(res["reward"])
        lengths.append(res["length"])
        
    env.close()  # Đóng environment
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_idx = int(np.argmin(rewards))
    max_idx = int(np.argmax(rewards))
    
    stats = {
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "best_episode": {"index": max_idx, "reward": rewards[max_idx], "length": lengths[max_idx]},
        "worst_episode": {"index": min_idx, "reward": rewards[min_idx], "length": lengths[min_idx]},
        "mean_length": float(np.mean(lengths))
    }
    return stats

def plot_results(rewards, window=20, save_path="/content/RL_22010846_LeThiThuy/Lab01/figures"):
    """Vẽ biểu đồ reward và moving average."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Reward từng Episode", color="gray")
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, color="blue", linewidth=2, label=f"Moving Average ({window} eps)")
        
    plt.title("Kết quả Huấn luyện/Đánh giá Agent trong CartPole-v1 (Main Experiment)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    fig_file = os.path.join(save_path, "main_experiment_results.png")
    plt.savefig(fig_file)
    plt.close()
    print(f"Đã lưu biểu đồ kết quả mới vào: {fig_file}")

def main():
    env_name = "CartPole-v1"
    n_episodes = 500
    seed = 42
    
    print(f"BẮT ĐẦU THÍ NGHIỆM RL TÊN MÔI TRƯỜNG: {env_name}")
    stats = evaluate_policy(env_name, policy, n_episodes=n_episodes, seed=seed)
    
    print("\n THỐNG KÊ KẾT QUẢ")
    print(f"Tổng số Episode           : {n_episodes}")
    print(f"Mean Reward               : {stats['mean_reward']:.2f}")
    print(f"Standard Deviation Reward : {stats['std_reward']:.2f}")
    print(f"Episode Tốt Nhất          : Ep {stats['best_episode']['index']} - Reward: {stats['best_episode']['reward']}")
    print(f"Episode Tệ Nhất           : Ep {stats['worst_episode']['index']} - Reward: {stats['worst_episode']['reward']}")
    print(f"Mean Episode Length       : {stats['mean_length']:.2f}")
    
    # Vẽ biểu đồ và lưu file với tên độc lập
    plot_results(stats["rewards"])
    
    # Kết luận
    print("\nKẾT LUẬN THÍ NGHIỆM")
    print("Policy heuristic cải tiến dựa trên cả góc nghiêng và vận tốc góc duy trì độ ổn định cao.")
    print("Môi trường được thiết lập seed giúp đảm bảo tính tái lập (reproducibility) cho thí nghiệm.")

if __name__ == "__main__":
    main()
