"""Bài 20: So sánh transition deterministic vs stochastic"""

from bai16 import create_environment

if __name__ == "__main__":
    state, action = 0, 2  # RIGHT
    env_det = create_environment(is_slippery=False)
    env_sto = create_environment(is_slippery=True)

    print("Deterministic:", env_det.unwrapped.P[state][action])
    print("Stochastic   :", env_sto.unwrapped.P[state][action])
    env_det.close(); env_sto.close()
