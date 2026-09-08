"""Bài 30: Kiểm tra policy stability"""

import numpy as np
from bai16 import create_environment
from mdp_utils import policy_evaluation, greedy_policy_matrix_from_value

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    gamma, theta = 0.99, 1e-8
    policy = np.ones((n_states, n_actions)) / n_actions
    i = 0
    while True:
        i += 1
        V, _ = policy_evaluation(env, policy, gamma, theta)
        new_policy = greedy_policy_matrix_from_value(env, V, gamma)
        policy_stable = np.array_equal(np.argmax(new_policy, axis=1), np.argmax(policy, axis=1))
        policy = new_policy
        if policy_stable:
            break
    print(f"Policy Iteration converged after {i} iterations.")
    env.close()
