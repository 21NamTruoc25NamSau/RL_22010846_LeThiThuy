"""Bài 34: Đánh giá policy bằng simulation"""

import numpy as np
from bai16 import create_environment
from mdp_utils import (value_iteration, policy_iteration,
                        greedy_policy_from_value, evaluate_policy_by_simulation)

if __name__ == "__main__":
    env = create_environment()
    n_states, n_actions = env.observation_space.n, env.action_space.n
    rng = np.random.default_rng(0)
    random_policy = rng.integers(0, n_actions, size=n_states)

    V_vi, _, _ = value_iteration(env, gamma=0.99, theta=1e-8)
    vi_policy = greedy_policy_from_value(env, V_vi, gamma=0.99)
    pi_policy, V_pi, _ = policy_iteration(env, gamma=0.99, theta=1e-8)

    for name, policy in [("Random", random_policy), ("Value Iteration", vi_policy), ("Policy Iteration", pi_policy)]:
        print(name, evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42))
    env.close()
