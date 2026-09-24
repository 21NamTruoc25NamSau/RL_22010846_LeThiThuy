"""Bài 29: So sánh epsilon"""
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import epsilon_greedy_action

def main():
    rng = np.random.default_rng(42)
    Q = defaultdict(lambda: np.zeros(2))
    state = "test_state"
    Q[state] = np.array([1.0, 5.0])

    epsilons = [0.01, 0.05, 0.10, 0.20, 0.50]
    n_samples = 10000
    greedy_freqs = []

    for eps in epsilons:
        actions = [epsilon_greedy_action(Q, state, 2, eps, rng) for _ in range(n_samples)]
        greedy_freqs.append(actions.count(1) / n_samples)

    fig_dir = os.path.join("Lab03", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot([str(e) for e in epsilons], greedy_freqs, marker='o', color='b')
    plt.title("Tần suất chọn Greedy Action theo Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Tần suất chọn Greedy Action")
    plt.grid(True)

    plt.savefig(os.path.join(fig_dir, "epsilon_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
