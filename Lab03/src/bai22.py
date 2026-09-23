"""Bài 22: Sai khác giữa hai phương pháp"""
import gymnasium as gym
import numpy as np
from mc_utils import first_visit_mc_prediction, every_visit_mc_prediction, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    n_episodes = 10000

    V_first, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_episodes)
    V_every, _ = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_episodes)

    common_states = set(V_first.keys()).intersection(set(V_every.keys()))
    diffs = [abs(V_first[st] - V_every[st]) for st in common_states]

    mean_abs_diff = np.mean(diffs)
    print(f"Mean absolute difference trên {len(common_states)} state chung: {mean_abs_diff:.6f}")

    env.close()

if __name__ == "__main__":
    main()
