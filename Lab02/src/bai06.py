"""Bài 06: So sánh phân phối lý thuyết và mô phỏng"""

FIG_PATH = 'c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\markov_distribution.png'

import numpy as np
import matplotlib.pyplot as plt
from bai01 import P, STATES
from bai04 import state_distribution
from bai05 import sample_next_state

N_TRANSITIONS = 100_000


def simulate_frequency(P, n_steps, start_state=0, seed=123):
    rng = np.random.default_rng(seed)
    state = start_state
    counts = np.zeros(len(P))
    for _ in range(n_steps):
        state = sample_next_state(state, P, rng)
        counts[state] += 1
    return counts / counts.sum()


if __name__ == "__main__":
    theoretical = state_distribution(np.array([1.0, 0.0, 0.0]), P, n_steps=50)
    empirical = simulate_frequency(P, N_TRANSITIONS)

    print("Lý thuyết :", theoretical)
    print("Mô phỏng  :", empirical)
    print("Sai lệch  :", np.abs(theoretical - empirical))

    x = np.arange(len(STATES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, theoretical, width, label="Lý thuyết (t=50)")
    ax.bar(x + width / 2, empirical, width, label="Mô phỏng (100000 bước)")
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_title("So sánh phân phối lý thuyết và mô phỏng - Markov Chain")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)

    plt.show()
