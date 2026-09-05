"""Bài 02: Kiểm tra tính hợp lệ của một transition matrix."""

import numpy as np
from bai01 import P


def validate_transition_matrix(P, tol=1e-10):
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if np.any(P < 0) or np.any(P > 1):
        return False
    row_sums = P.sum(axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) < tol))


if __name__ == "__main__":
    print("P hợp lệ:", validate_transition_matrix(P))
    bad_P = np.array([[0.5, 0.4], [0.3, 0.3]])
    print("bad_P hợp lệ:", validate_transition_matrix(bad_P))
