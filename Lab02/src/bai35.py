"""Bài 35: So sánh Value Iteration và Policy Iteration"""

FIG_PATH = "c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\algorithm_comparison.png"

from time import perf_counter
import matplotlib.pyplot as plt
from bai16 import create_environment
from mdp_utils import (value_iteration, policy_iteration,
                        greedy_policy_from_value, evaluate_policy_by_simulation)

if __name__ == "__main__":
    env = create_environment()
    t0 = perf_counter()
    V_vi, n_iter_vi, _ = value_iteration(env, gamma=0.99, theta=1e-8)
    t_vi = perf_counter() - t0
    vi_policy = greedy_policy_from_value(env, V_vi, gamma=0.99)
    stats_vi = evaluate_policy_by_simulation(env, vi_policy, n_episodes=1000, seed=42)

    t0 = perf_counter()
    pi_policy, V_pi, n_iter_pi = policy_iteration(env, gamma=0.99, theta=1e-8)
    t_pi = perf_counter() - t0
    stats_pi = evaluate_policy_by_simulation(env, pi_policy, n_episodes=1000, seed=42)

    print(f"{'Thuat toan':<18}{'Vong lap':>10}{'Thoi gian':>12}{'Success':>10}{'Reward':>10}")
    print(f"{'Value Iteration':<18}{n_iter_vi:>10}{t_vi:>12.5f}{stats_vi['success_rate']:>10.3f}{stats_vi['mean_reward']:>10.3f}")
    print(f"{'Policy Iteration':<18}{n_iter_pi:>10}{t_pi:>12.5f}{stats_pi['success_rate']:>10.3f}{stats_pi['mean_reward']:>10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Value Iter", "Policy Iter"], [n_iter_vi, n_iter_pi], color=["tab:green", "tab:blue"])
    axes[0].set_title("So vong lap")
    axes[1].bar(["Value Iter", "Policy Iter"], [t_vi, t_pi], color=["tab:green", "tab:blue"])
    axes[1].set_title("Thoi gian chay (s)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    env.close()
