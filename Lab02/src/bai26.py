"""Bài 26: Greedy policy từ value function V"""

import numpy as np
from bai16 import create_environment
from mdp_utils import greedy_policy_from_value, policy_evaluation

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    V, _ = policy_evaluation(env, policy, gamma=0.99)
    print("Greedy policy:", greedy_policy_from_value(env, V, gamma=0.99))
    env.close()
