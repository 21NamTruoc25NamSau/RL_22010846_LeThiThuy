"""Bài 11: Viết Random Agent"""

import gymnasium as gym

def random_agent(env, max_steps=500):
    env.reset()
    total_reward = 0.0
    episode_length = 0

    for _ in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1

        if terminated or truncated:
            break

    return total_reward, episode_length

def main():
    env = gym.make("CartPole-v1")
    total_reward, episode_length = random_agent(env)
    print("Total reward:", total_reward)
    print("Episode length:", episode_length)
    env.close()

if __name__ == "__main__":
    main()
