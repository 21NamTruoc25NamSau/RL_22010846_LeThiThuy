"""Bài 11: Policy cố định stick_on_20_policy"""
import os
import gymnasium as gym

def stick_on_20_policy(state):
    """
    State: (player_sum, dealer_card, usable_ace)
    Trả về:
      0 (Stick) nếu player_sum >= 20
      1 (Hit)   nếu player_sum < 20
    """
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0  # Stick
    return 1      # Hit

def main():
    env = gym.make("Blackjack-v1")
    state, _ = env.reset(seed=42)
    action = stick_on_20_policy(state)

    print(f"Kiểm tra Policy tại State {state}:")
    print(f"  -> Action được chọn: {action} ({'Stick' if action == 0 else 'Hit'})")

    # Cập nhật thêm policy này vào src/mc_utils.py để sử dụng ở các bài sau
    utils_path = os.path.join("src", "mc_utils.py")
    if os.path.exists(utils_path):
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write("\n\ndef stick_on_20_policy(state):\n    player_sum, dealer_card, usable_ace = state\n    return 0 if player_sum >= 20 else 1\n")

    env.close()

if __name__ == "__main__":
    main()
