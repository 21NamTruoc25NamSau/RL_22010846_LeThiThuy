"""Bài 29: Cài đặt Policy Iteration"""

from bai16 import create_environment
from mdp_utils import policy_iteration

if __name__ == "__main__":
    env = create_environment()
    policy, V, n_iterations = policy_iteration(env, gamma=0.99, theta=1e-8)
    print(f"Policy Iteration hội tụ sau {n_iterations} iterations")
    print("Optimal policy:", policy)
    env.close()
