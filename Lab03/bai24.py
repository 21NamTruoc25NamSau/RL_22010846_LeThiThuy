"""Bài 24: Lưu state-action-return"""
import gymnasium as gym
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    returns_sa = defaultdict(list)

    for ep_idx in range(100):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=1.0)

        for idx, step in enumerate(ep):
            st, action, _ = step
            returns_sa[(st, action)].append(g_list[idx])

    print(f"Số cặp (state, action) thu thập được: {len(returns_sa)}")
    sample_sa = list(returns_sa.keys())[0]
    print(f"{sample_sa}: {returns_sa[sample_sa][:5]}")
    env.close()

if __name__ == "__main__":
    main()
