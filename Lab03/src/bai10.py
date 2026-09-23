"""Bài 10: So sánh gamma và vẽ biểu đồ"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mc_utils import compute_returns

def main():
    env = gym.make("Blackjack-v1")
    gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
    n_episodes = 1000
    mean_g0 = []

    for gamma in gammas:
        g0_list = []
        for ep in range(n_episodes):
            state, _ = env.reset(seed=ep)
            terminated = False
            truncated = False
            rewards = []
            while not (terminated or truncated):
                action = env.action_space.sample()
                _, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
            g0_list.append(compute_returns(rewards, gamma)[0] if rewards else 0.0)
        mean_g0.append(np.mean(g0_list))

    print("Kết quả trung bình G_0 theo gamma:")
    for gamma, val in zip(gammas, mean_g0):
        print(f"Gamma = {gamma:<4}: Mean G_0 = {val:.4f}")

    # Tạo thư mục figures nếu chưa tồn tại
    fig_dir = os.path.join(".", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Vẽ biểu đồ so sánh Mean G_0 theo Gamma
    plt.figure(figsize=(8, 5))
    plt.plot([str(g) for g in gammas], mean_g0, marker='o', color='b', linewidth=2, markersize=8)
    plt.title("So sánh Mean G_0 theo các giá trị Gamma khác nhau")
    plt.xlabel("Hệ số chiết khấu (Gamma)")
    plt.ylabel("Giá trị trung bình G_0 (Mean G_0)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Lưu biểu đồ vào thư mục figures/
    output_fig_path = os.path.join(fig_dir, "gamma_comparison.png")
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ tại: {output_fig_path}")

    # Hiển thị biểu đồ khi chạy script
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
