"""Mở rộng 1: So sánh FrozenLake 4x4 và 8x8 bằng Value Iteration."""

from time import perf_counter
from bai16 import create_environment
from mdp_utils import value_iteration, greedy_policy_from_value, evaluate_policy_by_simulation

if __name__ == "__main__":
    print(f"{'Map':<6}{'So state':>10}{'So iteration':>15}{'Runtime (s)':>14}{'Success rate':>14}")
    for map_name in ["4x4", "8x8"]:
        env = create_environment(map_name=map_name, is_slippery=True)
        t0 = perf_counter()
        V, n_iter, _ = value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=100000)
        runtime = perf_counter() - t0
        policy = greedy_policy_from_value(env, V, gamma=0.99)
        stats = evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42)
        print(f"{map_name:<6}{env.observation_space.n:>10}{n_iter:>15}{runtime:>14.4f}{stats['success_rate']:>14.3f}")
        env.close()
