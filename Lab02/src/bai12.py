"""Bài 12: Xây dựng MDP hai state, hai action."""

P = {
    0: {
        0: [(0.8, 0, 0.0, False), (0.2, 1, 1.0, False)],
        1: [(0.3, 0, 0.0, False), (0.7, 1, 2.0, False)],
    },
    1: {
        0: [(0.6, 0, 1.0, False), (0.4, 1, 0.0, True)],
        1: [(1.0, 1, 5.0, True)],
    },
}
N_STATES, N_ACTIONS = 2, 2

if __name__ == "__main__":
    for s in P:
        for a in P[s]:
            print(f"State {s}, Action {a}: {P[s][a]}")
