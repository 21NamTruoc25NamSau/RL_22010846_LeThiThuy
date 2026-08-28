"""Bài 27: Reward trong FrozenLake"""
import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    total_episodes = 100
    success = 0
    failure = 0
    
    for _ in range(total_episodes):
        env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 1.0:
                    success += 1
                else:
                    failure += 1
                break
                
    env.close()
    
    success_rate = success / total_episodes
    
    print(f"Số episode thành công (success): {success}")
    print(f"Số episode thất bại (failure)  : {failure}")
    print(f"Success rate                   : {success_rate:.2%}")

if __name__ == "__main__":
    main()
