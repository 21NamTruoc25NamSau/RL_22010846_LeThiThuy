"""Bài 09: Tính return G_t cho từng bước"""

import numpy as np


def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


if __name__ == "__main__":
    rewards = [0, 0, 0, 1]
    G = discounted_returns(rewards, gamma=0.9)
    for t, g in enumerate(G):
        print(f"G_{t} = {g:.4f}")
