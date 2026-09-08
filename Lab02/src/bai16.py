"""Bài 16: Thông tin cơ bản của FrozenLake-v1."""

import gymnasium as gym


def create_environment(map_name="4x4", is_slippery=True):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)


if __name__ == "__main__":
    env = create_environment()
    obs, info = env.reset(seed=42)
    print("Number of states:", env.observation_space.n)
    print("Number of actions:", env.action_space.n)
    print("Initial observation:", obs)
    env.close()
