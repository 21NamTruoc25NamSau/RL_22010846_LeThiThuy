"""Bài 8: Kiểm tra return"""
from mc_utils import compute_returns

def main():
    rewards = [0, 0, 1]
    gammas = [1.0, 0.9, 0.5]
    print(f"Chuỗi reward mẫu: {rewards}")
    for gamma in gammas:
        ret = compute_returns(rewards, gamma)
        print(f"Gamma = {gamma:<4} -> Returns G_t: {ret}")

if __name__ == "__main__":
    main()
