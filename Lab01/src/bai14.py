"""Bài 14: Chạy 100 episode"""

import gymnasium as gym

def random_agent(env):
    env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main():
    env = gym.make("CartPole-v1")
    episode_rewards = []

    for _ in range(100):
        reward = random_agent(env)
        episode_rewards.append(reward)

    print(f"Đã hoàn thành 100 episode. Số lượng reward ghi nhận: {len(episode_rewards)}")
    env.close()

if __name__ == "__main__":
    main()
