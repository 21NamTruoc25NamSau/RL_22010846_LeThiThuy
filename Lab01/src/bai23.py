"""Bài 23: Tạo FrozenLake"""
import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    print("Observation Space:", env.observation_space)
    print("Action Space     :", env.action_space)
    
    # Xác định số state và số action
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    print(f"Số lượng state  : {n_states}")
    print(f"Số lượng action : {n_actions}")
    
    env.close()

if __name__ == "__main__":
    main()
