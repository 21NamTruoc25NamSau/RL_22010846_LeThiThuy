"""Bài 32: Value Iteration hoàn chỉnh"""

FIG_PATH = 'c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\value_iteration_convergence.png'

import matplotlib.pyplot as plt
from bai16 import create_environment
from mdp_utils import value_iteration

if __name__ == "__main__":
    env = create_environment()
    V, n_iterations, deltas = value_iteration(env, gamma=0.99, theta=1e-8)
    print(f"Value Iteration hội tụ sau {n_iterations} iterations")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(deltas)+1), deltas, color="tab:green")
    ax.set_yscale("log")
    ax.set_title("Hội tụ của Value Iteration")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Delta (log scale)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    env.close()
