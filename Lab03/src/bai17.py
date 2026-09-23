"""Bài 17: First-Visit MC Prediction"""
import os
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)

        visited_states = set()
        for idx, step in enumerate(ep):
            state = step[0]
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += g_list[idx]
                returns_count[state] += 1

    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count

def main():
    env = gym.make("Blackjack-v1")

    # 1. Gọi hàm và nhận lại V cùng returns_count
    V, returns_count = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=500)

    print(f"Tổng số states: {len(V)}")
    print(f"{'State (Player, Dealer, Ace)':<28} | {'V(s)':<10} | {'returns_count':<15}")
    print("-" * 60)

    # 2. In mẫu 10 trạng thái đầu tiên để kiểm tra dữ liệu trong V và returns_count
    for state in list(V.keys())[:10]:
        v_val = V[state]
        count_val = returns_count[state]
        print(f"{str(state):<28} | {v_val:<10.4f} | {count_val:<15d}")

    # Ghi nhận/cập nhật lại hàm vào src/mc_utils.py
    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):\n    returns_sum = defaultdict(float)\n    returns_count = defaultdict(int)\n    for ep_idx in range(n_episodes):\n        ep = generate_episode(env, policy=policy, seed=ep_idx)\n        rewards = [step[2] for step in ep]\n        g_list = compute_returns(rewards, gamma=gamma)\n        visited_states = set()\n        for idx, step in enumerate(ep):\n            state = step[0]\n            if state not in visited_states:\n                visited_states.add(state)\n                returns_sum[state] += g_list[idx]\n                returns_count[state] += 1\n    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}\n    return V, returns_count\n")

    env.close()

if __name__ == "__main__":
    main()
