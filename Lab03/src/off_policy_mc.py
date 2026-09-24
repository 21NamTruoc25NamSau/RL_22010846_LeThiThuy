"""Mở rộng 1: Off-policy Monte Carlo Control (Weighted Importance Sampling)"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import compute_returns, evaluate_policy

def create_behavior_policy(Q, epsilon, n_actions):
    def policy_probs(state):
        probs = np.ones(n_actions, dtype=float) * (epsilon / n_actions)
        best_action = np.argmax(Q[state]) if state in Q else 0
        probs[best_action] += (1.0 - epsilon)
        return probs
    return policy_probs

def off_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    C = defaultdict(lambda: np.zeros(n_actions))

    target_policy = lambda state: int(np.argmax(Q[state])) if state in Q else 0
    episode_rewards = []

    for ep_idx in range(n_episodes):
        behavior_policy = create_behavior_policy(Q, epsilon, n_actions)

        episode = []
        state, _ = env.reset(seed=int(rng.integers(0, 1e6)))
        done = False

        while not done:
            probs = behavior_policy(state)
            action = rng.choice(n_actions, p=probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        rewards = [step[2] for step in episode]
        episode_rewards.append(sum(rewards))

        G = 0.0
        W = 1.0

        for t in reversed(range(len(episode))):
            st, act, reward = episode[t]
            G = gamma * G + reward

            C[st][act] += W
            Q[st][act] += (W / C[st][act]) * (G - Q[st][act])

            if act != target_policy(st):
                break

            probs = behavior_policy(st)
            W = W * (1.0 / probs[act])

    return Q, target_policy, episode_rewards

def main():
    print("1. KHÁI NIỆM CỐT LÕI OFF-POLICY MONTE CARLO")
    print("   - Target Policy (pi): Chính sách mục tiêu cần tối ưu hóa (thường là Greedy policy).")
    print("   - Behavior Policy (b): Chính sách hành vi dùng để thám hiểm và thu thập dữ liệu (thường là Epsilon-greedy policy).")
    print("   - Importance Sampling: Kỹ thuật hiệu chỉnh trọng số xác suất khi học dữ liệu được tạo ra bởi một chính sách khác.")

    env = gym.make("Blackjack-v1")
    n_episodes = 50000
    print(f"\n2. Huấn luyện Off-policy MC Control với {n_episodes} episodes...")

    Q, target_policy, episode_rewards = off_policy_mc_control(env, n_episodes=n_episodes, gamma=1.0, epsilon=0.1, seed=42)
    print(f"   Đã học xong Q-table với {len(Q)} states.")

    print("\n3. Kiểm tra Target Policy thu được tại một số state mẫu:")
    for st in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
        act = target_policy(st)
        act_str = "Stick" if act == 0 else "Hit"
        print(f"   State {st}: Action = {act} ({act_str})")

    print("\n4. Đánh giá Target Policy (10,000 episodes):")
    res = evaluate_policy(env, target_policy, n_episodes=10000, seed=123)
    print(f"   Win Rate: {res['win_rate']:.4f} | Loss Rate: {res['loss_rate']:.4f} | Draw Rate: {res['draw_rate']:.4f} | Mean Reward: {res['mean_reward']:.4f}")

    fig_dir = os.path.join("Lab03", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    window = 1000
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg, color='purple')
    plt.title("Off-Policy MC Control Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Behavior Policy)")
    plt.grid(True)
    plot_path = os.path.join(fig_dir, "off_policy_mc_control.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n5. Training curve đã lưu tại: {plot_path}")

    env.close()

if __name__ == "__main__":
    main()
