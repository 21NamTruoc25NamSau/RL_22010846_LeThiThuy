"""Bài 10: Ảnh hưởng của gamma lên G_0"""

FIG_PATH = 'c:\\Users\\LENOVO\\Downloads\\Học Tăng Cường Cơ Bản\\Thực Hành\\Lab02\\figures\\gamma_comparison.png'

import numpy as np
import matplotlib.pyplot as plt
from bai07 import compute_return

rewards = [0, 0, 0, 0, 10]
gammas = np.linspace(0, 1, 101)

if __name__ == "__main__":
    returns = [compute_return(rewards, g) for g in gammas]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gammas, returns, color="tab:blue")
    ax.set_title("Ảnh hưởng của Discount Factor lên G_0")
    ax.set_xlabel("Gamma"); ax.set_ylabel("G_0 (Return)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print("G_0 tại gamma = 0:", returns[0], "| tại gamma = 1:", returns[-1])
