"""Bài 24: Hiển thị FrozenLake dạng text"""
import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, info = env.reset()
    
    grid_ascii = env.render()
    print("Bản đồ FrozenLake dạng Text (ANSI):")
    print(grid_ascii)
    
    # Quan sát vị trí:
    # S : Start (Vị trí bắt đầu)
    # F : Frozen (Băng an toàn)
    # H : Hole (Hố băng - Game Over)
    # G : Goal (Đích đến)
    
    env.close()

if __name__ == "__main__":
    main()
