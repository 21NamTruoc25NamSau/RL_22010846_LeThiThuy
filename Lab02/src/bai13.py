"""Bài 13: Kiểm tra MDP model."""

import numpy as np
from bai12 import P, N_STATES, N_ACTIONS


def validate_mdp(P, n_states, n_actions):
    ok = True
    for s in range(n_states):
        for a in range(n_actions):
            total_prob = sum(t[0] for t in P[s][a])
            if not np.isclose(total_prob, 1.0):
                print(f"Invalid transition at state={s}, action={a}")
                ok = False
    return ok


if __name__ == "__main__":
    print("MDP hợp lệ:", validate_mdp(P, N_STATES, N_ACTIONS))
