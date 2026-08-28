"""Bài 07. Một bước tương tác"""

import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    print("State before action:", observation)

    action = env.action_space.sample()
    print("Action:", action)

    observation, reward, terminated, truncated, info = env.step(action)

    print("State after action:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    env.close()

if __name__ == "__main__":
    main()
