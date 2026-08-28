"""Bài 26: Điều khiển FrozenLake bằng chuỗi action"""
import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, info = env.reset(seed=42)
    
    # Chuỗi hành động để đi từ S (0) tới G (15): Right, Right, Down, Down, Down, Right
    actions = [2, 2, 1, 1, 1, 2]
    
    print("Trạng thái khởi tạo:")
    print(env.render())
    
    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Bước {step} | Action: {action} ({ACTION_NAMES[action]}) | State hiện tại: {obs} | Reward: {reward}")
        print(env.render())
        if terminated or truncated:
            if reward == 1.0:
                print(">>> Đã đến Goal thành công!")
            else:
                print(">>> Agent đã rơi xuống hố!")
            break
            
    env.close()

if __name__ == "__main__":
    main()
