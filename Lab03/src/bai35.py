"""Bài 35: Learning curve"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mc_utils import on_policy_mc_control

def main():
    env = gym.make("Blackjack-v1")
    _, _, rewards = on_policy_mc_control(env, n_episodes=100000, seed=42)

    window = 1000
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

    fig_dir = os.path.join("Lab03", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='g', label=f"Moving Average (window={window})")
    plt.title("MC Control Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(fig_dir, "mc_convergence.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    env.close()

if __name__ == "__main__":
    main()
