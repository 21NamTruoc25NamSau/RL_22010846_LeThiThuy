"""Bài 15: Theo dõi sự hội tụ của một state cụ thể"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    target_state = (20, 10, False)
    checkpoints = [100, 500, 1000, 5000, 10000]

    returns = defaultdict(list)
    estimates = []

    print(f"Theo dõi trạng thái mục tiêu {target_state}:")

    for ep_idx in range(1, max(checkpoints) + 1):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=1.0)

        for idx, step in enumerate(ep):
            state = step[0]
            returns[state].append(g_list[idx])

        if ep_idx in checkpoints:
            v_val = np.mean(returns[target_state]) if len(returns[target_state]) > 0 else 0.0
            estimates.append(v_val)
            print(f"  Episode {ep_idx:5d}: V(state) = {v_val:.4f} (Số lượt thăm: {len(returns[target_state])})")

    # Lưu và vẽ biểu đồ hội tụ
    fig_dir = os.path.join(".", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot([str(cp) for cp in checkpoints], estimates, marker='s', color='g', linewidth=2)
    plt.title(f"Đường hội tụ V(s) cho State {target_state}")
    plt.xlabel("Số lượng Episodes")
    plt.ylabel("Ước lượng V(s)")
    plt.grid(True, linestyle="--", alpha=0.7)

    output_path = os.path.join(fig_dir, "mc_convergence.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ hội tụ tại: {output_path}")
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
