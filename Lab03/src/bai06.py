"""Bài 6: Chuỗi reward"""
import gymnasium as gym
from mc_utils import generate_episode

def main():
    env = gym.make("Blackjack-v1")
    episode = generate_episode(env, seed=2026)
    rewards = [step[2] for step in episode]
    print("Độ dài Episode:", len(episode))
    print("Danh sách Rewards:", rewards)
    print("Tổng Reward (Total Reward):", sum(rewards))
    env.close()

if __name__ == "__main__":
    main()
