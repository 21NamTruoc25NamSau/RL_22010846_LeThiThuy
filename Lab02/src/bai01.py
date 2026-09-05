"""Bài 01: Tạo transition matrix cho Markov chain thời tiết (Sunny, Cloudy, Rainy)."""

import numpy as np

STATES = ["Sunny", "Cloudy", "Rainy"]

P = np.array([
    [0.7, 0.2, 0.1],   # từ Sunny
    [0.3, 0.4, 0.3],   # từ Cloudy
    [0.2, 0.3, 0.5],   # từ Rainy
])


def main():
    print("Transition matrix P (hàng = state hiện tại, cột = state kế tiếp):")
    print(P)
    print("Tổng mỗi hàng:", P.sum(axis=1))


if __name__ == "__main__":
    main()
