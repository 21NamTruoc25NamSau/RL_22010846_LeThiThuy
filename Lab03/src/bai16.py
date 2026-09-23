"""Bài 16: Phát hiện first visit trong episode"""
import gymnasium as gym
from mc_utils import generate_episode

def get_first_visit_indices(episode):
    """
    Trả về dict dạng {state: first_index}
    chứa vị trí xuất hiện ĐẦU TIÊN của mỗi state trong episode.
    """
    first_visits = {}
    for idx, step in enumerate(episode):
        state = step[0]
        if state not in first_visits:
            first_visits[state] = idx
    return first_visits

def main():
    env = gym.make("Blackjack-v1")
    # Sinh 1 episode mẫu
    ep = generate_episode(env, seed=42)

    print("Các bước trong episode (state, action, reward):")
    for idx, step in enumerate(ep):
        print(f"  Bước {idx}: State = {step[0]}")

    fv_map = get_first_visit_indices(ep)
    print("\nKết quả vị trí First-Visit của từng State:")
    for st, f_idx in fv_map.items():
        print(f"  State {st} xuất hiện lần đầu ở bước {f_idx}")

    env.close()

if __name__ == "__main__":
    main()
