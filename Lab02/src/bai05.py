"""Bài 05: Mô phỏng Markov chain"""

import numpy as np
from bai01 import P, STATES


def sample_next_state(current_state, P, rng):
    return rng.choice(len(P), p=P[current_state])


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    state = 0
    trajectory = [state]
    for _ in range(30):
        state = sample_next_state(state, P, rng)
        trajectory.append(state)
    print("Chuỗi state (tên):", [STATES[s] for s in trajectory])
