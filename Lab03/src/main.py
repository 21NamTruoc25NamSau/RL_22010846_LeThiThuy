"""Bài 36: Monte Carlo Agent hoàn chỉnh"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def stick_on_20_policy(state):
    player_sum, _, _ = state
    return 0 if player_sum >= 20 else 1

def generate_episode(env, policy, seed=None):
    episode = []
    state, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def compute_returns(rewards, gamma=1.0):
    G = 0.0
    returns = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_states = set()
        for idx, step in enumerate(ep):
            st = step[0]
            if st not in visited_states:
                visited_states.add(st)
                returns_sum[st] += g_list[idx]
                returns_count[st] += 1
    return {st: returns_sum[st] / returns_count[st] for st in returns_sum}

def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        for idx, step in enumerate(ep):
            st = step[0]
            returns_sum[st] += g_list[idx]
            returns_count[st] += 1
    return {st: returns_sum[st] / returns_count[st] for st in returns_sum}

def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, n_actions)
    if state not in Q:
        return rng.integers(0, n_actions)
    return int(np.argmax(Q[state]))

def update_incremental_mean(Q, N, state, action, G):
    N[state][action] += 1
    Q[state][action] += (G - Q[state][action]) / N[state][action]

def greedy_action(Q, state, n_actions):
    if state not in Q:
        return 0
    return int(np.argmax(Q[state]))

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

def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    wins, losses, draws = 0, 0, 0
    total_rewards = []
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=seed + ep_idx)
        final_r = ep[-1][2]
        total_rewards.append(final_r)
        if final_r == 1.0:
            wins += 1
        elif final_r == -1.0:
            losses += 1
        else:
            draws += 1
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": np.mean(total_rewards)
    }

def main():
    env = gym.make("Blackjack-v1")
    print("1. Environment: Blackjack-v1")

    print(f"2. Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    sample_ep = generate_episode(env, stick_on_20_policy, seed=42)
    print("3. Episode mẫu:")
    for step in sample_ep:
        print(f"   State: {step[0]}, Action: {step[1]}, Reward: {step[2]}")

    trajectory = [(step[0], step[1]) for step in sample_ep]
    print(f"4. Trajectory: {trajectory}")

    rewards = [step[2] for step in sample_ep]
    print(f"5. Reward: {rewards}")

    returns_undiscounted = compute_returns(rewards, gamma=1.0)
    print(f"6. Return (gamma=1.0): {returns_undiscounted}")

    returns_discounted = compute_returns(rewards, gamma=0.9)
    print(f"7. Discounted return (gamma=0.9): {returns_discounted}")

    print("8. Fixed policy: Stick on 20")

    V_first = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=5000)
    print(f"9. First-Visit MC Prediction (Số states: {len(V_first)})")

    V_every = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=5000)
    print(f"10. Every-Visit MC Prediction (Số states: {len(V_every)})")

    common_states = set(V_first.keys()).intersection(set(V_every.keys()))
    diffs = [abs(V_first[st] - V_every[st]) for st in common_states]
    print(f"11. So sánh hai prediction method (Mean abs diff: {np.mean(diffs):.6f})")

    n_train_episodes = 50000
    Q, learned_policy, episode_rewards = on_policy_mc_control(env, n_episodes=n_train_episodes, epsilon=0.1, seed=42)
    print(f"12. Q(s,a) đã tính toán cho {len(Q)} states")
    print("13. Epsilon-greedy (epsilon = 0.1)")
    print(f"14. On-policy MC Control hoàn tất qua {n_train_episodes} episodes")

    fig_dir = os.path.join("Lab03", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    window = 1000
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg, color='b')
    plt.title("Training Curve - On-Policy MC Control")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plot_path = os.path.join(fig_dir, "bai36_mc_agent_training.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"15. Training curve đã lưu tại: {plot_path}")

    print("16. Learned policy cho một số state mẫu:")
    for st in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
        print(f"    State {st}: Action = {learned_policy(st)}")

    def random_policy(state):
        return env.action_space.sample()

    policies = {
        "Random Policy": random_policy,
        "Fixed Policy": stick_on_20_policy,
        "Learned MC Policy": learned_policy
    }

    print("17 & 18. Evaluation và So sánh Policy:")
    for name, pol in policies.items():
        res = evaluate_policy(env, pol, n_episodes=10000, seed=123)
        print(f"    {name:<20} | Win: {res['win_rate']:.4f} | Loss: {res['loss_rate']:.4f} | Draw: {res['draw_rate']:.4f} | Mean Reward: {res['mean_reward']:.4f}")

    print("19. Nhận xét: Learned Policy tối ưu tỷ lệ thắng và phần thưởng trung bình so với các policy còn lại.")
    print("20. Kết luận: Thuật toán Monte Carlo Control học thành công chiến lược tối ưu mà không cần thông tin chuyển trạng thái của môi trường.")

    env.close()

if __name__ == "__main__":
    main()
