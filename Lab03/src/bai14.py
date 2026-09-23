"""Bài 14: Ước lượng V(s) bằng trung bình mẫu"""
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    returns = defaultdict(list)
    n_episodes = 2000

    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=1.0)

        for idx, step in enumerate(ep):
            state = step[0]
            returns[state].append(g_list[idx])

    # Tính V(s)
    V = {state: np.mean(ret) for state, ret in returns.items()}

    print(f"Đã tính toán V(s) cho {len(V)} trạng thái.")
    print("\nGiá trị V(s) của 10 trạng thái ngẫu nhiên:")
    for state in list(V.keys())[:10]:
        print(f"  State {str(state):<18}: V(s) = {V[state]:.4f} (Số lượt mẫu: {len(returns[state])})")

    env.close()

if __name__ == "__main__":
    main()
