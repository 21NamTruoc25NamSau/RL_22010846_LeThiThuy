"""Bài 27: Hiển thị policy trên lưới 4x4"""

import numpy as np
from bai16 import create_environment
from mdp_utils import print_frozenlake_policy, policy_evaluation, greedy_policy_from_value

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    V, _ = policy_evaluation(env, policy, gamma=0.99)
    greedy_policy = greedy_policy_from_value(env, V, gamma=0.99)
    print_frozenlake_policy(env, greedy_policy)
    env.close()
