"""
Bài 02: Khởi tạo môi trường CartPole.
"""
import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    print("Environment object:", env)
    env.close()

if __name__ == "__main__":
    main()
