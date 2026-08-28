"""Bài 06: Sinh action ngẫu nhiên."""

import gymnasium as gym
from collections import Counter

def main():
    env = gym.make("CartPole-v1")

    # Sinh 20 action bằng sample() và lưu vào danh sách
    actions = [env.action_space.sample() for _ in range(20)]

    # In toàn bộ danh sách
    print("Danh sách 20 action:", actions)

    # Tính và in tần suất xuất hiện của từng action
    frequencies = dict(Counter(actions))
    print("Tần suất xuất hiện từng action:", frequencies)

    env.close()

if __name__ == "__main__":
    main()
