"""Bài 17: In transition model của state=0."""

from bai16 import create_environment

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

if __name__ == "__main__":
    env = create_environment()
    P = env.unwrapped.P
    for action in range(env.action_space.n):
        print(f"\nAction {action} ({ACTION_NAMES[action]}):")
        for probability, next_state, reward, terminated in P[0][action]:
            print(f"  p={probability:.3f}, next_state={next_state}, reward={reward}, terminated={terminated}")
    env.close()
