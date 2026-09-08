"""Bài 24: Iterative Policy Evaluation"""

import numpy as np
from bai16 import create_environment
from mdp_utils import policy_evaluation

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    V, n_iterations = policy_evaluation(env, policy, gamma=0.99, theta=1e-8)
    print(f"Hội tụ sau {n_iterations} iterations")
    print(V.reshape(4, 4))
    env.close()
