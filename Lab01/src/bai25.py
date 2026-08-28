"""Bài 25: Ánh xạ action"""
import gymnasium as gym

# Định nghĩa dictionary ánh xạ action
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset(seed=42)
    
    # Sinh một action ngẫu nhiên
    action = env.action_space.sample()
    action_name = ACTION_NAMES[action]
    
    print(f"Action {action} -> {action_name}")
    env.close()

if __name__ == "__main__":
    main()
