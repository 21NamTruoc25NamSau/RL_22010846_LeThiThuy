"""Bài 33: Học policy cho Blackjack"""
import gymnasium as gym
from mc_utils import on_policy_mc_control

def main():
    env = gym.make("Blackjack-v1")
    Q, policy, _ = on_policy_mc_control(env, n_episodes=100000, seed=42)

    test_states = [
        (20, 10, False),
        (18, 6, False),
        (13, 2, False),
        (18, 6, True)
    ]

    print("Chính sách học được tại các state ví dụ:")
    for st in test_states:
        act = policy(st)
        act_str = "Stick" if act == 0 else "Hit"
        print(f"  State {str(st):<20}: Action = {act} ({act_str})")

    env.close()

if __name__ == "__main__":
    main()
