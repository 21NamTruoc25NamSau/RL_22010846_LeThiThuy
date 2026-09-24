"""Bài 27: Epsilon-greedy policy"""
import os
import numpy as np
from collections import defaultdict

def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, n_actions)
    else:
        if state not in Q:
            return rng.integers(0, n_actions)
        return int(np.argmax(Q[state]))

def main():
    rng = np.random.default_rng(42)
    Q = defaultdict(lambda: np.zeros(2))
    Q[(15, 10, False)] = np.array([0.1, 0.9])

    act = epsilon_greedy_action(Q, (15, 10, False), 2, 0.1, rng)
    print(f"Action được chọn từ epsilon-greedy: {act}")

    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef epsilon_greedy_action(Q, state, n_actions, epsilon, rng):\n    import numpy as np\n    if rng.random() < epsilon:\n        return rng.integers(0, n_actions)\n    else:\n        if state not in Q:\n            return rng.integers(0, n_actions)\n        return int(np.argmax(Q[state]))\n")

if __name__ == "__main__":
    main()
