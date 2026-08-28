"""Bài 16: Episode tốt nhất"""

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
    history = []

    for ep in range(1, 101):
        reward, length = random_agent(env)
        history.append((ep, reward, length))
    env.close()

    best_episode = max(history, key=lambda x: x[1])

    print(f"Episode tốt nhất: {best_episode[0]}")
    print(f"Reward tương ứng: {best_episode[1]:.2f}")
    print(f"Độ dài episode tương ứng: {best_episode[2]}")

if __name__ == "__main__":
    main()
