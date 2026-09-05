"""Bài 15: Stochastic policy (uniform random)."""

import numpy as np
from bai12 import N_STATES, N_ACTIONS

policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

if __name__ == "__main__":
    print("Stochastic policy:\n", policy)
    print("Tổng mỗi hàng:", policy.sum(axis=1))
