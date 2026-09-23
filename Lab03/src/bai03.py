"""Bài 3: Phân tích action"""
import gymnasium as gym

ACTION_NAMES = {
    0: "Stick (Dằn bài / Ngưng rút)",
    1: "Hit (Rút thêm bài)"
}

def main():
    env = gym.make("Blackjack-v1")
    print("Action Space:", env.action_space)
    print("Số lượng hành động khả thi:", env.action_space.n)
    print("Danh sách hành động:")
    for action_id, action_name in ACTION_NAMES.items():
        print(f"  ID {action_id}: {action_name}")
    env.close()

if __name__ == "__main__":
    main()
