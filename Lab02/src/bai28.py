"""Bài 28: Một bước Policy Improvement."""

import numpy as np
from bai16 import create_environment
from mdp_utils import policy_evaluation, greedy_policy_from_value

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    old_policy_matrix = np.ones((n_states, n_actions)) / n_actions
    old_policy = np.argmax(old_policy_matrix, axis=1)

    V, _ = policy_evaluation(env, old_policy_matrix, gamma=0.99)
    new_policy = greedy_policy_from_value(env, V, gamma=0.99)

    print(f"Số state đổi action: {np.sum(old_policy != new_policy)}/{n_states}")
    env.close()
