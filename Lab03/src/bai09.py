"""Bài 9: Gắn return vào trajectory"""
import gymnasium as gym
from mc_utils import generate_episode, compute_returns

def main():
    env = gym.make("Blackjack-v1")
    raw_episode = generate_episode(env, seed=15)
    rewards = [item[2] for item in raw_episode]
    returns = compute_returns(rewards, gamma=0.95)

    full_trajectory = [
        (raw_episode[i][0], raw_episode[i][1], raw_episode[i][2], returns[i])
        for i in range(len(raw_episode))
    ]

    print("Trajectory chứa Return (state, action, reward, G_t):")
    for step in full_trajectory:
        print(f"  {step}")
    env.close()

if __name__ == "__main__":
    main()
