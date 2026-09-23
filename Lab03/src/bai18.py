"""Bài 18: Đánh giá số lượng state theo số episode"""
import gymnasium as gym
from mc_utils import first_visit_mc_prediction, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    episodes_list = [100, 1000, 10000, 50000]

    print("Thống kê số lượng state đã được ước lượng:")
    for n_ep in episodes_list:
        V, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_ep)
        print(f"  Số episodes = {n_ep:5d} -> Số state đã ước lượng được: {len(V):3d}")

    env.close()

if __name__ == "__main__":
    main()
