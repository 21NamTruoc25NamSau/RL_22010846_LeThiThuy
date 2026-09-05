"""Bài 08: Discounted return"""

from bai07 import compute_return

rewards = [1, 1, 1, 1, 1]
gammas = [0.0, 0.5, 0.9, 0.99, 1.0]

if __name__ == "__main__":
    print(f"{'Gamma':>6} | {'Return':>10}")
    print("-" * 20)
    for g in gammas:
        print(f"{g:6.2f} | {compute_return(rewards, g):10.4f}")
