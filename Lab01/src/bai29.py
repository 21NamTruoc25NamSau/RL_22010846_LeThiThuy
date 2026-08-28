"""Bài 29: Viết policy dưới dạng hàm"""
import gymnasium as gym

def policy(observation, env):
    # Ban đầu policy chỉ trả về action ngẫu nhiên
    return env.action_space.sample()

def run_agent_with_policy(env):
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        # Thay thế env.action_space.sample() bằng policy(observation)
        action = policy(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main():
    env = gym.make("CartPole-v1")
    reward = run_agent_with_policy(env)
    print(f"Total reward thu được với random policy dạng hàm: {reward}")
    env.close()

if __name__ == "__main__":
    main()
