"""Bài 30: Khởi tạo Q"""
import gymnasium as gym
import numpy as np
from collections import defaultdict

def main():
    env = gym.make("Blackjack-v1")
    n_actions = env.action_space.n

    # Cách 1: Sử dụng returns_sum và returns_count
    Q_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    # Cách 2: Sử dụng incremental mean (cần Q và N đếm số lần)
    Q_inc = defaultdict(lambda: np.zeros(n_actions))
    N_count = defaultdict(lambda: np.zeros(n_actions))

    sample_state = (15, 10, False)

    print("--- CÁCH 1: Dùng returns_sum và returns_count ---")
    print(f"Khởi tạo Q_sum[{sample_state}]: {Q_sum[sample_state]}")
    print(f"Khởi tạo returns_sum[({sample_state}, action=0)]: {returns_sum[(sample_state, 0)]}")
    print(f"Khởi tạo returns_count[({sample_state}, action=0)]: {returns_count[(sample_state, 0)]}\n")

    print("--- CÁCH 2: Dùng incremental mean (Q và N) ---")
    print(f"Khởi tạo Q_inc[{sample_state}]: {Q_inc[sample_state]}")
    print(f"Khởi tạo N_count[{sample_state}]: {N_count[sample_state]}")

    env.close()

if __name__ == "__main__":
    main()
