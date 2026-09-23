"""Bài 5: Lưu episode"""
import gymnasium as gym
from mc_utils import generate_episode

def main():
    env = gym.make("Blackjack-v1")
    ep = generate_episode(env, seed=10)
    print("Trajectory thu được (state, action, reward):")
    for idx, step in enumerate(ep):
        print(f"  Bước {idx+1}: {step}")
    env.close()

if __name__ == "__main__":
    main()
