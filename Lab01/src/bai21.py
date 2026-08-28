"""Bài 21: Seed cho action space"""
import gymnasium as gym

def generate_actions(seed):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    
    # Thiết lập seed cho action_space
    env.action_space.seed(seed)
    
    actions = [env.action_space.sample() for _ in range(20)]
    env.close()
    return actions

def main():
    actions_run1 = generate_actions(seed=42)
    actions_run2 = generate_actions(seed=42)
    
    print("Lần 1:", actions_run1)
    print("Lần 2:", actions_run2)
    print("Chuỗi action hai lần chạy có giống nhau không:", actions_run1 == actions_run2)

if __name__ == "__main__":
    main()
