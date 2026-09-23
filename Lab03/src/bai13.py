"""Bài 13: Thu thập return theo state"""
import gymnasium as gym
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    returns = defaultdict(list)
    n_episodes = 500

    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=1.0)

        for idx, step in enumerate(ep):
            state = step[0]
            returns[state].append(g_list[idx])

    print(f"Đã thu thập dữ liệu qua {n_episodes} episode.")
    print(f"Tổng số trạng thái duy nhất đã xuất hiện: {len(returns)}")
    sample_state = list(returns.keys())[0]
    print(f"Ví dụ trạng thái {sample_state} đã thu thập {len(returns[sample_state])} lượt return. 5 return đầu: {returns[sample_state][:5]}")
    env.close()

if __name__ == "__main__":
    main()
