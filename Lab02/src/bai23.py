"""Bài 23: Một sweep của Policy Evaluation."""

import numpy as np
from bai16 import create_environment
from mdp_utils import policy_evaluation_sweep

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)
    new_V = policy_evaluation_sweep(env, policy, V, gamma=0.99)
    print("V sau 1 sweep:", new_V)
    env.close()
