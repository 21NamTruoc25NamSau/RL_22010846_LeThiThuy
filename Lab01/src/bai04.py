"""Bài 04: Khám phá observation space."""

import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")

    # In trực tiếp observation space
    print("Observation space:", env.observation_space)

    # Các thông số chi tiết
    obs_space = env.observation_space
    print("Shape của observation:", obs_space.shape)
    print("Kiểu dữ liệu:", obs_space.dtype)
    print("Giới hạn dưới:", obs_space.low)
    print("Giới hạn trên:", obs_space.high)

    env.close()

if __name__ == "__main__":
    main()
