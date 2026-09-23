"""Bài 12: Sinh 100 episode với stick_on_20_policy"""
import gymnasium as gym
from mc_utils import generate_episode, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    n_episodes = 100
    wins = 0
    losses = 0
    draws = 0

    for seed in range(n_episodes):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=seed)
        final_reward = ep[-1][2]  # Phần thưởng ở bước cuối cùng
        if final_reward == 1.0:
            wins += 1
        elif final_reward == -1.0:
            losses += 1
        else:
            draws += 1

    print(f"Kết quả sau {n_episodes} episode:")
    print(f"  Thắng (Win) : {wins:3d} ({wins/n_episodes*100:.1f}%)")
    print(f"  Thua  (Loss): {losses:3d} ({losses/n_episodes*100:.1f}%)")
    print(f"  Hòa   (Draw): {draws:3d} ({draws/n_episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    main()
