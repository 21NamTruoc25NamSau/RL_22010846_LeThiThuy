"""Bài 05: Quan sát trạng thái ban đầu."""

import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    print("Observation:", observation)
    print("Type:", type(observation))
    print("Shape:", observation.shape)
    print("Info:", info)

    # Kiểu dữ liệu của từng phần tử trong observation:
    # observation[0] (Cart Position): float32
    # observation[1] (Cart Velocity): float32
    # observation[2] (Pole Angle): float32
    # observation[3] (Pole Angular Velocity): float32

    env.close()

if __name__ == "__main__":
    main()
