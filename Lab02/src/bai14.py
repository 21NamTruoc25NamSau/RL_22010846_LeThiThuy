"""Bài 14: Deterministic policy"""

import numpy as np

policy = np.array([1, 0])  # State 0 -> Action 1, State 1 -> Action 0


def print_policy(policy):
    for s, a in enumerate(policy):
        print(f"State {s} -> Action {a}")


if __name__ == "__main__":
    print_policy(policy)
