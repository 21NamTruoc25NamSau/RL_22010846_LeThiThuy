"""Bài 21: Một Bellman backup - tính Q(s,a)."""

import numpy as np
from bai16 import create_environment
from mdp_utils import q_from_v

if __name__ == "__main__":
    env = create_environment()
    V = np.zeros(env.observation_space.n)
    q = q_from_v(env, V, state=14, action=1, gamma=0.99)
    print("Q(state=14, action=DOWN) với V=0:", q)
    env.close()
