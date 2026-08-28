"""
Bài 03: Khám phá không gian hành động.
"""
import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    print("Action space:", env.action_space)

    num_actions = env.action_space.n
    print("Number of actions:", num_actions)

    env.close()

if __name__ == "__main__":
    main()
