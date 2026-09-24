"""Bài 26: Greedy action từ Q"""
import os
import numpy as np
from collections import defaultdict

def greedy_action(Q, state, n_actions):
    if state not in Q:
        return 0
    return int(np.argmax(Q[state]))

def main():
    Q = defaultdict(lambda: np.zeros(2))
    Q[(20, 10, False)] = np.array([0.8, 0.2])

    act = greedy_action(Q, (20, 10, False), 2)
    print(f"Greedy action được chọn: {act}")

    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef greedy_action(Q, state, n_actions):\n    import numpy as np\n    if state not in Q:\n        return 0\n    return int(np.argmax(Q[state]))\n")

if __name__ == "__main__":
    main()
