"""Bài 31: Incremental mean"""
import os
import numpy as np
from collections import defaultdict

def update_incremental_mean(Q, N, state, action, G):
    N[state][action] += 1
    Q[state][action] += (G - Q[state][action]) / N[state][action]

def main():
    Q = defaultdict(lambda: np.zeros(2))
    N = defaultdict(lambda: np.zeros(2))

    st, act, G = (15, 10, False), 1, 1.0
    update_incremental_mean(Q, N, st, act, G)
    print(f"Cập nhật lần 1: Q = {Q[st]}, N = {N[st]}")

    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef update_incremental_mean(Q, N, state, action, G):\n    N[state][action] += 1\n    Q[state][action] += (G - Q[state][action]) / N[state][action]\n")

if __name__ == "__main__":
    main()
