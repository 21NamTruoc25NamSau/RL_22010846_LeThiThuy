"""Bài 33: Xây dựng hàm run_episode()"""
import gymnasium as gym

def run_episode(env, policy, seed=None, max_steps=1000):
    """
    Hàm tổng quát chạy 1 episode trên môi trường bất kỳ.
    Returns:
        dict: {"reward": total_reward, "length": length, "terminated": terminated, "truncated": truncated}
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False
    
    for step in range(max_steps):
        # Đảm bảo hàm policy nhận tham số phù hợp
        try:
            action = policy(obs, env)
        except TypeError:
            action = policy(obs)
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        
        if terminated or truncated:
            break
            
    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated
    }

def random_policy(observation, env):
    return env.action_space.sample()

def main():
    env = gym.make("CartPole-v1")
    result = run_episode(env, random_policy, seed=42)
    print("Kết quả chạy 1 episode:")
    print(result)
    env.close()

if __name__ == "__main__":
    main()
