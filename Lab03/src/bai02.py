"""Bài 2: Phân tích observation"""
import gymnasium as gym

# Môi trường Blackjack-v1 trả về observation là một tuple gồm 3 thành phần:
# 1. player_sum (int): Tổng điểm hiện tại của người chơi (thường từ 4 đến 21 điểm).
# 2. dealer_card (int): Lá bài ngửa của nhà cái (dealer) đang mở trên bàn (từ 1 đến 10, trong đó 1 đại diện cho con Ách/Ace).
# 3. usable_ace (bool/int): Cờ đánh dấu người chơi có con Ách linh hoạt (Usable Ace) hay không:
#    - 1 (hoặc True): Người chơi sở hữu ít nhất 1 lá Ách đang được tính là 11 điểm mà tổng điểm không vượt quá 21.
#    - 0 (hoặc False): Người chơi không có lá Ách linh hoạt (không có Ách hoặc Ách bắt buộc phải tính là 1 điểm).

def main():
    env = gym.make("Blackjack-v1")
    print("In 10 observation từ 10 episode ngẫu nhiên:")
    for i in range(10):
        obs, _ = env.reset(seed=i)

        # Giải nén tuple observation thành 3 biến tương ứng
        player_sum, dealer_card, usable_ace = obs

        print(f"Episode {i+1:2d}: Player Sum = {player_sum:2d} | Dealer Card = {dealer_card:2d} | Usable Ace = {usable_ace}")
    env.close()

if __name__ == "__main__":
    main()
