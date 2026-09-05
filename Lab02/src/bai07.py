"""Bài 07: Undiscounted return"""


def compute_return(rewards, gamma):
    G = 0.0
    for k, r in enumerate(rewards):
        G += (gamma ** k) * r
    return G


if __name__ == "__main__":
    rewards = [1, 1, 1, 1, 1]
    print("rewards =", rewards)
    print("Undiscounted return G (gamma=1.0) =", compute_return(rewards, gamma=1.0))
