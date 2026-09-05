"""Bài 03: Tính xác suất trạng thái kế tiếp"""

import numpy as np
from bai01 import P

p0 = np.array([1.0, 0.0, 0.0])

if __name__ == "__main__":
    p1 = p0 @ P
    print("p0 =", p0)
    print("p1 =", p1)
    print("Tổng p1 =", p1.sum())
