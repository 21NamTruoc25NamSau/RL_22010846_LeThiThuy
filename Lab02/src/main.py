"""Bài 36: Mini-project - Dynamic Programming Solver cho FrozenLake-v1."""

FIG_PATH = "c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\dp_solver_convergence.png"

import argparse
from time import perf_counter
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import (
    ACTION_NAMES, policy_evaluation, greedy_policy_from_value,
    policy_iteration, value_iteration, evaluate_policy_by_simulation,
    print_frozenlake_policy,
)


def create_environment(map_name="4x4", is_slippery=True):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)


def print_policy(env, policy, title="Policy"):
    print(f"\n{title}:", policy)
    print_frozenlake_policy(env, policy)


def plot_convergence(deltas_dict, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, deltas in deltas_dict.items():
        ax.plot(range(1, len(deltas)+1), deltas, label=label)
    ax.set_yscale("log")
    ax.set_title("Convergence cua Dynamic Programming")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Delta (log)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-8)
    parser.add_argument("--n_episodes", type=int, default=1000)
    args, _ = parser.parse_known_args()

    env = create_environment()
    t0 = perf_counter()
    V_vi, n_iter_vi, deltas_vi = value_iteration(env, args.gamma, args.theta)
    t_vi = perf_counter() - t0
    vi_policy = greedy_policy_from_value(env, V_vi, args.gamma)
    print_policy(env, vi_policy, "Optimal Policy (Value Iteration)")

    t0 = perf_counter()
    pi_policy, V_pi, n_iter_pi = policy_iteration(env, args.gamma, args.theta)
    t_pi = perf_counter() - t0
    print_policy(env, pi_policy, "Optimal Policy (Policy Iteration)")

    stats_vi = evaluate_policy_by_simulation(env, vi_policy, args.n_episodes, seed=42)
    stats_pi = evaluate_policy_by_simulation(env, pi_policy, args.n_episodes, seed=42)

    print(f"\nVI: iters={n_iter_vi}, t={t_vi:.4f}s, success={stats_vi['success_rate']:.3f}")
    print(f"PI: iters={n_iter_pi}, t={t_pi:.4f}s, success={stats_pi['success_rate']:.3f}")

    plot_convergence({"Value Iteration": deltas_vi}, FIG_PATH)
    env.close()


if __name__ == "__main__":
    main()
