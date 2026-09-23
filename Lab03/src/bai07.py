"""Bài 7: Tính return từ cuối episode"""
from mc_utils import compute_returns

def main():
    sample_rewards = [0, 0, 1]
    gamma = 0.9
    g_list = compute_returns(sample_rewards, gamma=gamma)
    print("Chuỗi Rewards mẫu:", sample_rewards)
    print(f"Returns thu được (gamma={gamma}):", g_list)

if __name__ == "__main__":
    main()
