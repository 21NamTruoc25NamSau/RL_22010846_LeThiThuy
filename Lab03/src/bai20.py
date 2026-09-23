"""Bài 20: Every-Visit MC Prediction"""
import os
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, first_visit_mc_prediction, stick_on_20_policy

def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)

        for idx, step in enumerate(ep):
            state = step[0]
            returns_sum[state] += g_list[idx]
            returns_count[state] += 1

    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count

def main():
    env = gym.make("Blackjack-v1")
    n_ep = 5000

    V_fv, count_fv = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_ep)
    V_ev, count_ev = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_ep)

    print(f"So sánh sau {n_ep} episodes:")
    sample_state = (14, 8, False)
    print(f"  State {sample_state}:")
    print(f"    - First-Visit : V(s) = {V_fv.get(sample_state, 0):.4f} | Số lượt tính = {count_fv.get(sample_state, 0)}")
    print(f"    - Every-Visit : V(s) = {V_ev.get(sample_state, 0):.4f} | Số lượt tính = {count_ev.get(sample_state, 0)}")

    # Ghi nhận hàm này vào mc_utils.py
    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):\n    returns_sum = defaultdict(float)\n    returns_count = defaultdict(int)\n    for ep_idx in range(n_episodes):\n        ep = generate_episode(env, policy=policy, seed=ep_idx)\n        rewards = [step[2] for step in ep]\n        g_list = compute_returns(rewards, gamma=gamma)\n        for idx, step in enumerate(ep):\n            state = step[0]\n            returns_sum[state] += g_list[idx]\n            returns_count[state] += 1\n    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}\n    return V, returns_count\n")

    env.close()

if __name__ == "__main__":
    main()
