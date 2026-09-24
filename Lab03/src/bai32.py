"""Bài 32: On-policy First-Visit MC Control"""
import os
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import compute_returns, epsilon_greedy_action, update_incremental_mean, greedy_action

def generate_episode_control(env, Q, epsilon, rng):
    episode = []
    state, _ = env.reset()
    done = False
    n_actions = env.action_space.n

    while not done:
        action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    episode_rewards = []

    for ep_idx in range(n_episodes):
        ep = generate_episode_control(env, Q, epsilon, rng)
        rewards = [step[2] for step in ep]
        episode_rewards.append(sum(rewards))
        g_list = compute_returns(rewards, gamma=gamma)

        visited_sa = set()
        for idx, step in enumerate(ep):
            st, act, _ = step
            sa = (st, act)
            if sa not in visited_sa:
                visited_sa.add(sa)
                update_incremental_mean(Q, N, st, act, g_list[idx])

    def policy(state):
        return greedy_action(Q, state, n_actions)

    return Q, policy, episode_rewards

def main():
    env = gym.make("Blackjack-v1")
    n_episodes = 5000

    # Gọi hàm và nhận đúng 3 giá trị trả về
    Q, policy, episode_rewards = on_policy_mc_control(env, n_episodes=n_episodes, gamma=1.0, epsilon=0.1, seed=42)

    print("=== KIỂM TRA CÁC GIÁ TRỊ TRẢ VỀ CỦA HÀM ===")

    # 1. Kiểm tra Q
    print(f"1. [Q]: Đã học được hàm Q-value cho {len(Q)} states.")
    sample_st = (18, 6, False)
    print(f"   Ví dụ Q tại state {sample_st}: {Q[sample_st]}")

    # 2. Kiểm tra policy
    print(f"\n2. [policy]: Hàm policy trả về hành động greedy dựa trên Q.")
    test_action = policy(sample_st)
    act_str = "Stick" if test_action == 0 else "Hit"
    print(f"   Hành động tối ưu tại {sample_st} thu được từ policy(state): Action {test_action} ({act_str})")

    # 3. Kiểm tra episode_rewards
    print(f"\n3. [episode_rewards]: Danh sách phần thưởng qua từng episode (Độ dài = {len(episode_rewards)}).")
    print(f"   Reward của 5 episode đầu : {episode_rewards[:5]}")
    print(f"   Reward của 5 episode cuối: {episode_rewards[-5:]}")
    print(f"   Trung bình reward 500 episode cuối: {np.mean(episode_rewards[-500:]):.4f}")

    # Lưu hàm bổ trợ vào src/mc_utils.py
    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef generate_episode_control(env, Q, epsilon, rng):\n    episode = []\n    state, _ = env.reset()\n    done = False\n    n_actions = env.action_space.n\n    while not done:\n        action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)\n        next_state, reward, terminated, truncated, _ = env.step(action)\n        done = terminated or truncated\n        episode.append((state, action, reward))\n        state = next_state\n    return episode\n\ndef on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):\n    import numpy as np\n    from collections import defaultdict\n    rng = np.random.default_rng(seed)\n    n_actions = env.action_space.n\n    Q = defaultdict(lambda: np.zeros(n_actions))\n    N = defaultdict(lambda: np.zeros(n_actions))\n    episode_rewards = []\n    for ep_idx in range(n_episodes):\n        ep = generate_episode_control(env, Q, epsilon, rng)\n        rewards = [step[2] for step in ep]\n        episode_rewards.append(sum(rewards))\n        g_list = compute_returns(rewards, gamma=gamma)\n        visited_sa = set()\n        for idx, step in enumerate(ep):\n            st, act, _ = step\n            sa = (st, act)\n            if sa not in visited_sa:\n                visited_sa.add(sa)\n                update_incremental_mean(Q, N, st, act, g_list[idx])\n    def policy(state):\n        return greedy_action(Q, state, n_actions)\n    return Q, policy, episode_rewards\n")

    env.close()

if __name__ == "__main__":
    main()
