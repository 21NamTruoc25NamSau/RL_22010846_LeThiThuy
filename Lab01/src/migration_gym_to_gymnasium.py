"""
Chuyển đổi code từ OpenAI Gym cũ sang Gymnasium mới.
"""
import gymnasium as gym

def main():
    # Khởi tạo môi trường với Gymnasium
    env = gym.make("CartPole-v1")
    
    # reset() trả về tuple (observation, info) thay vì chỉ observation như Gym cũ
    observation, info = env.reset(seed=42)
    
    for t in range(1000):
        action = env.action_space.sample()
        
        # step() trả về 5 giá trị (observation, reward, terminated, truncated, info)
        # Thay vì 4 giá trị (observation, reward, done, info) như Gym cũ.
        observation, reward, terminated, truncated, info = env.step(action)
        
        # GIẢI THÍCH CHI TIẾT:
        #
        # 1. terminated có ý nghĩa gì?
        #    - terminated = True khi agent đạt tới trạng thái kết thúc tự nhiên của bài toán/môi trường
        #      (ví dụ: xe làm gậy đổ trong CartPole, hoặc agent đi tới Goal/rơi xuống Hole trong FrozenLake).
        #
        # 2. truncated có ý nghĩa gì?
        #    - truncated = True khi episode bị cắt ngang do đạt giới hạn thời gian / số bước quy định 
        #      (ví dụ: vượt quá max_episode_steps=500), chứ agent chưa hẳn đã thất bại hay đạt đích.
        #
        # 3. Vì sao không nên dùng done của API cũ?
        #    - Trạng thái 'done' ở API cũ gộp chung cả 2 trường hợp (terminated và truncated).
        #    - Việc tách rời 2 khái niệm này rất quan trọng trong Reinforcement Learning (đặc biệt là tính toán V-value / Q-value):
        #      + Nếu terminated: Trạng thái tiếp theo là trạng thái kết thúc thật sự (Bootstrapping V(s') = 0).
        #      + Nếu truncated: Trạng thái vẫn tiếp diễn nhưng bị ngắt hạn ngạch, vẫn cần ước lượng V(s') cho bước tiếp theo.
        
        # Episode kết thúc khi terminated OR truncated
        if terminated or truncated:
            print(f"Episode kết thúc tại bước {t+1}. Terminated: {terminated}, Truncated: {truncated}")
            break

    env.close()

if __name__ == "__main__":
    main()
