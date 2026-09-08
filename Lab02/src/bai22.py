"""Bài 22: Tính toàn bộ Q(s,.) cho một state."""

import numpy as np
from bai16 import create_environment
from mdp_utils import action_values

if __name__ == "__main__":
    env = create_environment()
    V = np.zeros(env.observation_space.n)
    for s in [0, 14]:
        print(f"State {s}: Q values = {action_values(env, V, s, gamma=0.99)}")
    env.close()
