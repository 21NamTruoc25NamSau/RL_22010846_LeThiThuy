**HỌC PHẦN:** HỌC TĂNG CƯỜNG CƠ BẢN - REINFORCEMENT LEARNING

**Sinh viên thực hiện:** Lê Thị Thúy
**Mã sinh viên:** 22010846

## Mục tiêu bài thực hành

Bài thực hành này được thiết kế để giới thiệu và củng cố kiến thức về các khái niệm cơ bản trong Học tăng cường (Reinforcement Learning - RL) và cách làm việc với thư viện **Gymnasium** – một công cụ tiêu chuẩn để phát triển và đánh giá các thuật toán RL. Các mục tiêu chính bao gồm:

*   **Cài đặt và kiểm tra môi trường:** Đảm bảo các thư viện cần thiết như Gymnasium, NumPy, Matplotlib hoạt động ổn định.
*   **Khám phá môi trường Gymnasium:** Hiểu rõ cấu trúc `action_space`, `observation_space`, cách thức hoạt động của `reset()` và `step()`.
*   **Tương tác Agent – Environment:** Thực hiện các bước tương tác đơn lẻ, chạy nhiều timestep, tính tổng reward và độ dài episode.
*   **Phân biệt `terminated` và `truncated`:** Nắm vững ý nghĩa và tầm quan trọng của hai trạng thái kết thúc episode trong API mới của Gymnasium.
*   **Thống kê và trực quan hóa dữ liệu:** Tính toán các chỉ số thống kê (mean, std, min, max reward) và vẽ biểu đồ thể hiện hiệu năng của agent qua các episode, bao gồm cả moving average.
*   **Tầm quan trọng của Random Seed:** Chứng minh tính tái lập (reproducibility) của thí nghiệm khi sử dụng seed.
*   **Làm quen với môi trường rời rạc FrozenLake:** Hiểu cách hoạt động của môi trường dạng lưới, điều khiển agent bằng chuỗi hành động và so sánh hiệu năng giữa môi trường deterministic và stochastic.
*   **Phát triển Policy:** Xây dựng và so sánh các loại policy khác nhau (Random, Always-Left/Right, Angle-based, Improved Heuristic) trên môi trường CartPole-v1, từ đó thấy được sự cải thiện rõ rệt về hiệu năng khi policy tận dụng thông tin từ observation.

## Cấu trúc bài thực hành

Bài thực hành được chia thành các phần chính như sau:

*   **Phần A: Cài đặt và khám phá Gymnasium**
    *   Kiểm tra môi trường Python và phiên bản Gymnasium.
    *   Tạo và khám phá không gian hành động (`action_space`) và không gian quan sát (`observation_space`) của môi trường CartPole-v1.
    *   Quan sát trạng thái ban đầu và sinh hành động ngẫu nhiên.
*   **Phần B: Tương tác Agent – Environment**
    *   Tìm hiểu về một bước tương tác (`step()`) và các giá trị trả về (`observation`, `reward`, `terminated`, `truncated`, `info`).
    *   Chạy nhiều bước tương tác và tính tổng phần thưởng (`total_reward`) cùng độ dài episode (`episode_length`).
    *   Xây dựng `random_agent` và hiểu sự khác biệt giữa `terminated` và `truncated` trong API mới của Gymnasium.
*   **Phần C: Episode và thống kê thực nghiệm**
    *   Chạy nhiều episode (10, 100) để thu thập dữ liệu về `reward`.
    *   Tính toán các chỉ số thống kê (mean, min, max, std) của reward.
    *   Xác định episode có reward tốt nhất.
    *   Vẽ biểu đồ reward theo episode và sử dụng `moving average` để làm mượt dữ liệu.
*   **Phần D: Random seed và khả năng tái lập**
    *   Thực hiện thí nghiệm với `seed` để hiểu về tính tái lập (`reproducibility`) của các kết quả.
    *   So sánh kết quả khi sử dụng các `seed` khác nhau cho môi trường và `action_space`.
    *   Xây dựng hàm thí nghiệm tổng quát (`experiment`) để tự động hóa việc này.
*   **Phần E: Làm quen môi trường rời rạc FrozenLake**
    *   Tạo và khám phá môi trường FrozenLake-v1.
    *   Hiển thị môi trường dạng text và ánh xạ các hành động.
    *   Điều khiển agent bằng chuỗi hành động được xác định trước.
    *   Đánh giá `success rate` của `random policy` trong FrozenLake.
    *   So sánh hiệu năng giữa môi trường `deterministic` (không trượt) và `stochastic` (có trượt).
*   **Phần F: Policy và cải thiện agent**
    *   Định nghĩa `policy` dưới dạng hàm.
    *   Xây dựng và so sánh các policy đơn giản (luôn trái, luôn phải).
    *   Phát triển `angle-based policy` dựa trên `observation` và `improved heuristic policy` kết hợp nhiều thành phần của `observation` để cải thiện hiệu suất.
*   **Phần G: Tổ chức code như một thí nghiệm RL**
    *   Xây dựng các hàm chung `run_episode()` và `evaluate_policy()` để tổ chức code một cách có hệ thống.
    *   Thực hiện một thí nghiệm hoàn chỉnh để so sánh hiệu năng của ba loại agent (Random, Angle-based, Improved) trên môi trường CartPole-v1, bao gồm bảng thống kê và biểu đồ so sánh.
    *   Mini-project cuối cùng để tổng hợp kiến thức.

## Hướng dẫn chạy bài thực hành

Để chạy bài thực hành này, bạn có thể thực hiện các bước sau:

1.  **Clone Repository:**
    Đầu tiên, bạn cần clone repository này về máy cục bộ hoặc mở trực tiếp trong Google Colab.
    ```bash
    git clone https://github.com/your-username/RL_22010846_LeThiThuy.git # Thay `your-username` bằng tên người dùng GitHub của bạn
    cd RL_22010846_LeThiThuy/Lab01
    ```

2.  **Cài đặt môi trường:**
    Bài thực hành này được xây dựng trên Python 3 và sử dụng thư viện `Gymnasium` cùng với `NumPy` và `Matplotlib`. Bạn có thể cài đặt các thư viện cần thiết bằng cách chạy các lệnh sau trong terminal hoặc trong các ô code của Jupyter/Colab:

    ```bash
    !pip install gymnasium==1.3.0
    !pip install gymnasium[classic-control,toy-text]==1.3.0
    !pip install numpy matplotlib jupyter
    ```
    _Lưu ý: Nếu bạn đang sử dụng Google Colab, các lệnh `!pip install` đã được cung cấp sẵn trong các ô code đầu tiên của notebook._

3.  **Chạy Notebook:**
    Mở tệp notebook chính (`RL_22010846_LeThiThuy.ipynb`) trong Jupyter Notebook, JupyterLab hoặc Google Colab.

    **Thực hiện các ô code theo thứ tự từ trên xuống dưới.** Mỗi ô code và ô văn bản (markdown) đã được sắp xếp để dẫn dắt bạn qua từng phần của bài thực hành. Đảm bảo chạy tất cả các ô code để thấy được kết quả và các biểu đồ minh họa.

    *   **Kết quả sẽ hiển thị ngay bên dưới mỗi ô code** sau khi thực thi. Các biểu đồ sẽ được lưu vào thư mục `figures/` trong cùng thư mục `Lab01`.

## Kết luận chung

Bài thực hành này cung cấp một nền tảng vững chắc để hiểu về Học tăng cường, từ việc cài đặt môi trường, khám phá các thành phần cơ bản của môi trường RL, đến việc thiết kế và đánh giá các chính sách (policy) đơn giản. Nó nhấn mạnh tầm quan trọng của việc sử dụng `random seed` để đảm bảo tính tái lập của thí nghiệm và cách tận dụng thông tin từ `observation` để cải thiện hiệu suất của agent.

Các kiến thức và kỹ năng thu được từ bài thực hành này sẽ là hành trang quan trọng cho các bài tập phức tạp hơn trong lĩnh vực Reinforcement Learning.
'''
