"""Bài 09: Chạy 20 bước"""

import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    env.reset(seed=42)

    for t in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"t: {t}, action: {action}, reward: {reward}")

        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    main()
