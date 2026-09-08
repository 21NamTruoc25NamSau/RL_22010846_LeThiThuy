"""Bài 31: Một sweep của Value Iteration"""

import numpy as np
from bai16 import create_environment
from mdp_utils import value_iteration_sweep

if __name__ == "__main__":
    env = create_environment()
    V = np.zeros(env.observation_space.n)
    print("V sau 1 sweep:", value_iteration_sweep(env, V, gamma=0.99))
    env.close()
