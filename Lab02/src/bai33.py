"""Bài 33: Trích xuất optimal policy sau Value Iteration"""

from bai16 import create_environment
from mdp_utils import value_iteration, greedy_policy_from_value, print_frozenlake_policy

if __name__ == "__main__":
    env = create_environment()
    V, n_iterations, _ = value_iteration(env, gamma=0.99, theta=1e-8)
    optimal_policy = greedy_policy_from_value(env, V, gamma=0.99)
    print("Optimal policy:", optimal_policy)
    print_frozenlake_policy(env, optimal_policy)
    env.close()
