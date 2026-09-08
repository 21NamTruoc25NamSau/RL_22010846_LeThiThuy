"""Bài 25: Theo dõi hội tụ Policy Evaluation"""

FIG_PATH = 'c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\policy_iteration_convergence.png'

import numpy as np
import matplotlib.pyplot as plt
from bai16 import create_environment
from mdp_utils import policy_evaluation_with_history

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    V, n_iterations, deltas = policy_evaluation_with_history(env, policy, gamma=0.99, theta=1e-8)
    print(f"Hội tụ sau {n_iterations} iterations")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(deltas)+1), deltas, color="tab:orange")
    ax.set_yscale("log")
    ax.set_title("Hội tụ của Iterative Policy Evaluation")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Delta (log scale)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    env.close()
