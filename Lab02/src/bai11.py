"""Bài 11: So sánh reward sớm và reward trễ"""

import numpy as np
from bai07 import compute_return

sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]

if __name__ == "__main__":
    gammas = np.linspace(0, 1, 1001)
    returns_A = np.array([compute_return(sequence_A, g) for g in gammas])
    returns_B = np.array([compute_return(sequence_B, g) for g in gammas])

    B_greater = gammas[returns_B > returns_A]
    if len(B_greater) > 0:
        print(f"B > A khi gamma trong khoảng [{B_greater.min():.3f}, {B_greater.max():.3f}]")

    diff = returns_B - returns_A
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    for idx in sign_change:
        print(f"Điểm giao xấp xỉ gamma = {gammas[idx]:.4f}")
