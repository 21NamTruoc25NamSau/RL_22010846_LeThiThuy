"""Bài 21: So sánh First-Visit và Every-Visit"""
import gymnasium as gym
from mc_utils import first_visit_mc_prediction, every_visit_mc_prediction, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    n_episodes = 10000
    gamma = 1.0

    V_first, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_episodes, gamma=gamma)
    V_every, _ = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_episodes, gamma=gamma)

    common_states = list(set(V_first.keys()).intersection(set(V_every.keys())))[:10]

    print(f"{'State':<25} | {'V_first(s)':<12} | {'V_every(s)':<12}")
    print("-" * 55)
    for st in common_states:
        print(f"{str(st):<25} | {V_first[st]:<12.4f} | {V_every[st]:<12.4f}")

    env.close()

if __name__ == "__main__":
    main()
