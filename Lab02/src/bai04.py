"""Bài 04: Phân phối trạng thái sau n bước"""

import numpy as np
from bai01 import P
from bai03 import p0


def state_distribution(p0, P, n_steps):
    p = np.array(p0, dtype=float)
    for _ in range(n_steps):
        p = p @ P
    return p


if __name__ == "__main__":
    for t in [1, 2, 5, 10, 50]:
        print(f"t={t:2d}: {state_distribution(p0, P, t)}")
