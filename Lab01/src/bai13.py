"""Bài 13: Chạy 10 episode"""

import gymnasium as gym

def random_agent(env):
    env.reset()
    total_reward = 0.0
    episode_length = 0
    while True:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_length += 1
        if terminated or truncated:
            break
    return total_reward, episode_length

def main():
    env = gym.make("CartPole-v1")
    print(f"{'Episode':<10} | {'Reward':<10} | {'Length':<10}")
    print("-" * 36)
    for ep in range(1, 11):
        reward, length = random_agent(env)
        print(f"{ep:<10} | {reward:<10.1f} | {length:<10}")
    env.close()

if __name__ == "__main__":
    main()
