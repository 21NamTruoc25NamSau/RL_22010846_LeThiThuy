"""Bài 25: Ước lượng Q(s,a)"""
import os
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)

        visited_sa = set()
        for idx, step in enumerate(ep):
            st, action, _ = step
            sa = (st, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                returns_sum[sa] += g_list[idx]
                returns_count[sa] += 1

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for (st, action), total_r in returns_sum.items():
        Q[st][action] = total_r / returns_count[(st, action)]

    return Q

def main():
    env = gym.make("Blackjack-v1")
    n_episodes = 10000
    Q = mc_action_value_prediction(env, stick_on_20_policy, n_episodes=n_episodes)

    # In giá trị Q[state][action] cụ thể cho một số state ví dụ
    sample_states = [(20, 10, False), (18, 6, False), (13, 2, False)]

    print(f"Tổng số state thu thập được Q: {len(Q)}\n")
    print(f"{'State':<20} | {'Q(s, Action=0/Stick)':<22} | {'Q(s, Action=1/Hit)':<20}")
    print("-" * 68)

    for st in sample_states:
        q_stick = Q[st][0]  # Q[state][0]
        q_hit = Q[st][1]    # Q[state][1]
        print(f"{str(st):<20} | {q_stick:<22.4f} | {q_hit:<20.4f}")

    # Cập nhật hàm vào mc_utils.py
    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):\n    from collections import defaultdict\n    import numpy as np\n    returns_sum = defaultdict(float)\n    returns_count = defaultdict(int)\n    for ep_idx in range(n_episodes):\n        ep = generate_episode(env, policy=policy, seed=ep_idx)\n        rewards = [step[2] for step in ep]\n        g_list = compute_returns(rewards, gamma=gamma)\n        visited_sa = set()\n        for idx, step in enumerate(ep):\n            st, action, _ = step\n            sa = (st, action)\n            if sa not in visited_sa:\n                visited_sa.add(sa)\n                returns_sum[sa] += g_list[idx]\n                returns_count[sa] += 1\n    Q = defaultdict(lambda: np.zeros(env.action_space.n))\n    for (st, action), total_r in returns_sum.items():\n        Q[st][action] = total_r / returns_count[(st, action)]\n    return Q\n")

    env.close()

if __name__ == "__main__":
    main()
