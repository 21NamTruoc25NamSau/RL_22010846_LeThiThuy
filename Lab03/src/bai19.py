"""Bài 19: Visualize value"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mc_utils import first_visit_mc_prediction, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    V, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=50000)

    # Tạo ma trận giá trị cho Usable Ace = False
    # Player sum: 12 -> 21 (10 dòng), Dealer showing card: 1 -> 10 (10 cột)
    value_grid = np.zeros((10, 10))

    for p_sum in range(12, 22):
        for d_card in range(1, 11):
            state = (p_sum, d_card, False)
            value_grid[p_sum - 12, d_card - 1] = V.get(state, 0.0)

    # Vẽ Heatmap
    fig_dir = os.path.join(".", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(value_grid, xticklabels=range(1, 11), yticklabels=range(12, 22), 
                cmap="viridis", annot=True, fmt=".2f")
    plt.title("Hàm giá trị V(s) với Usable Ace = False (Stick-on-20 Policy)")
    plt.xlabel("Lá bài ngửa của Dealer (Dealer Showing Card)")
    plt.ylabel("Tổng điểm người chơi (Player Sum)")

    output_path = os.path.join(fig_dir, "policy_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu Heatmap tại: {output_path}")
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
