"""Bài 08: Viết hàm run_on_step()"""

import gymnasium as gym

def run_one_step(env, action):
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info

def main():
    env = gym.make("CartPole-v1")
    env.reset(seed=42)

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = run_one_step(env, action)
        print(f"Step {i+1} | Action: {action} | Obs: {obs} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    main()
