"""Bài 18: Hàm describe_state()"""

from bai16 import create_environment
from bai17 import ACTION_NAMES


def describe_state(env, state):
    P = env.unwrapped.P
    print(f"=== State {state} ===")
    for action in range(env.action_space.n):
        print(f"  Action {action} ({ACTION_NAMES[action]}):")
        for probability, next_state, reward, terminated in P[state][action]:
            print(f"    p={probability:.3f}, next_state={next_state}, reward={reward}, terminated={terminated}")


if __name__ == "__main__":
    env = create_environment()
    for s in [0, 1, 14]:
        describe_state(env, s)
    env.close()
