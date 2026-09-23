"""Bài 23: Biểu đồ hội tụ"""
import os
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import generate_episode, compute_returns, stick_on_20_policy

def main():
    env = gym.make("Blackjack-v1")
    target_states = [(20, 10, False), (18, 6, False), (13, 2, False)]
    checkpoints = [100, 500, 1000, 2000, 5000, 10000]

    first_history = {st: [] for st in target_states}
    every_history = {st: [] for st in target_states}

    first_sum, first_count = defaultdict(float), defaultdict(int)
    every_sum, every_count = defaultdict(float), defaultdict(int)

    for ep_idx in range(1, max(checkpoints) + 1):
        ep = generate_episode(env, policy=stick_on_20_policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=1.0)

        visited_first = set()
        for idx, step in enumerate(ep):
            st = step[0]
            every_sum[st] += g_list[idx]
            every_count[st] += 1

            if st not in visited_first:
                visited_first.add(st)
                first_sum[st] += g_list[idx]
                first_count[st] += 1

        if ep_idx in checkpoints:
            for st in target_states:
                v_f = first_sum[st] / first_count[st] if first_count[st] > 0 else 0.0
                v_e = every_sum[st] / every_count[st] if every_count[st] > 0 else 0.0
                first_history[st].append(v_f)
                every_history[st].append(v_e)

    fig_dir = os.path.join("Lab03", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))
    for i, st in enumerate(target_states):
        plt.subplot(1, 3, i + 1)
        plt.plot(checkpoints, first_history[st], label="First-Visit", marker='o')
        plt.plot(checkpoints, every_history[st], label="Every-Visit", marker='s', linestyle='--')
        plt.title(f"State: {st}")
        plt.xlabel("Episodes")
        plt.ylabel("V(s)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "first_vs_every_visit.png"), dpi=300)
    plt.close()
    env.close()

if __name__ == "__main__":
    main()
