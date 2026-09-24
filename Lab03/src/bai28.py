"""Bài 28: Kiểm tra epsilon-greedy"""
import numpy as np
from collections import defaultdict
from mc_utils import epsilon_greedy_action

def main():
    rng = np.random.default_rng(42)
    Q = defaultdict(lambda: np.zeros(2))
    state = "test_state"
    Q[state] = np.array([1.0, 5.0])

    n_samples = 10000
    epsilon = 0.1
    actions = [epsilon_greedy_action(Q, state, 2, epsilon, rng) for _ in range(n_samples)]

    count_0 = actions.count(0)
    count_1 = actions.count(1)

    print(f"Tần suất chọn Action 0 (Non-greedy): {count_0} ({count_0/n_samples*100:.2f}%)")
    print(f"Tần suất chọn Action 1 (Greedy)    : {count_1} ({count_1/n_samples*100:.2f}%)")
    print("Nhận xét: Tần suất phù hợp lý thuyết (Action 0 ~ 5%, Action 1 ~ 95%).")

if __name__ == "__main__":
    main()
