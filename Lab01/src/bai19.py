"""Bài 19: Thử nghiệm với seed"""
import gymnasium as gym
import numpy as np

def main():
    observations = []
    
    for _ in range(10):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=42)
        observations.append(obs)
        env.close()
        
    all_equal = all(np.array_equal(observations[0], obs) for obs in observations)
    print("Các initial observation có giống nhau không:", all_equal)
    print("Initial observation mẫu:", observations[0])

    # Kết luận:
    # 1. Việc truyền tham số seed=42 vào env.reset() giúp cố định trạng thái khởi tạo ngẫu nhiên của môi trường.
    # 2. Tất cả 10 môi trường độc lập đều trả về cùng một giá trị observation ban đầu giống nhau tuyệt đối.
    # 3. Điều này đảm bảo tính tái lập (reproducibility) trong các thử nghiệm Học tăng cường.

if __name__ == "__main__":
    main()
