"""Bài 34: Đánh giá policy đã học"""
import os
import gymnasium as gym
import numpy as np
from mc_utils import on_policy_mc_control, stick_on_20_policy, generate_episode

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

    def random_policy(state):
        return env.action_space.sample()

    _, learned_policy, _ = on_policy_mc_control(env, n_episodes=100000, seed=42)

    policies = {
        "Random Policy": random_policy,
        "Fixed Policy (Stick on 20)": stick_on_20_policy,
        "MC Learned Policy": learned_policy
    }

    print(f"{'Policy':<28} | {'Win Rate':<10} | {'Loss Rate':<10} | {'Draw Rate':<10} | {'Mean Reward':<11}")
    print("-" * 78)

    for name, pol in policies.items():
        res = evaluate_policy(env, pol, n_episodes=10000, seed=123)
        print(f"{name:<28} | {res['win_rate']:<10.4f} | {res['loss_rate']:<10.4f} | {res['draw_rate']:<10.4f} | {res['mean_reward']:<11.4f}")

    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef evaluate_policy(env, policy, n_episodes=10000, seed=123):\n    import numpy as np\n    wins, losses, draws = 0, 0, 0\n    total_rewards = []\n    for ep_idx in range(n_episodes):\n        ep = generate_episode(env, policy=policy, seed=seed + ep_idx)\n        final_r = ep[-1][2]\n        total_rewards.append(final_r)\n        if final_r == 1.0:\n            wins += 1\n        elif final_r == -1.0:\n            losses += 1\n        else:\n            draws += 1\n    return {\n        'win_rate': wins / n_episodes,\n        'loss_rate': losses / n_episodes,\n        'draw_rate': draws / n_episodes,\n        'mean_reward': np.mean(total_rewards)\n    }\n")

    env.close()

if __name__ == "__main__":
    main()
