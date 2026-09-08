"""Bài 19: Kiểm tra tổng xác suất transition của mọi (state, action)."""

import numpy as np
from bai16 import create_environment

if __name__ == "__main__":
    env = create_environment()
    P = env.unwrapped.P
    all_ok = True
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            probs = [t[0] for t in P[s][a]]
            if not np.isclose(sum(probs), 1.0):
                all_ok = False
                print(f"LỖI tại state={s}, action={a}")
    print("Tất cả (state, action) có tổng xác suất = 1:", all_ok)
    env.close()
