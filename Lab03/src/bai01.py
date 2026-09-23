"""Bài 1: Tạo Blackjack-v1"""
import gymnasium as gym

def main():
    env = gym.make("Blackjack-v1")
    obs, info = env.reset(seed=42)
    print("Observation ban đầu:", obs)
    print("Info:", info)
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    env.close()

if __name__ == "__main__":
    main()
