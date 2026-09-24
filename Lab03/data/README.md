# Lab03 - Monte Carlo Methods

## Thông tin sinh viên

- Họ tên: Lê Thị Thuý
- MSSV: 22010846
- Lớp: Kỹ Thuật Robot Và Trí Tuệ Nhân Tạo (K16)
- Học phần: Học Tăng Cường Cơ Bản (N01)
- GitHub username: 21namtruoc25namsau
- 
## Cấu trúc thư mục

```
Lab03/
├── README.md
├── requirements.txt
├── src/
│ ├── bai01.py
│ ├── bai02.py
│ ├── ...
│ ├── bai36.py
│ ├── mc_utils.py
│ └── main.py
├── notebooks/
│ └── Lab03_22010846_LeThiThuy.ipynb
├── figures/
│ ├── mc_convergence.png
│ ├── first_vs_every_visit.png
│ ├── epsilon_comparison.png
│ ├── policy_performance.png
└── data/
  └── README.md
```

## Môi trường đã chọn

Chương trình sử dụng môi trường **Blackjack-v1** từ thư viện Gymnasium:
- **Observation Space (Trạng thái):** Tuple gồm 3 phần tử `(Player sum, Dealer showing card, Usable ace)`
  - `Player sum`: Tổng điểm của người chơi (từ 4 đến 21).
  - `Dealer showing card`: Lá bài ngửa của nhà cái (từ 1 đến 10).
  - `Usable ace`: Có sử dụng lá Ách tính 11 điểm hay không (`True`/`False`).
- **Action Space (Hành động):** 
  - `0`: Stick (Dừng bài).
  - `1`: Hit (Rút tiếp lá bài).

## Mục tiêu

- Nắm vững lý thuyết cốt lõi về phương pháp Monte Carlo trong Reinforcement Learning (RL).
- Triển khai thuật toán **MC Prediction** (First-Visit và Every-Visit) để ước lượng giá trị trạng thái $V(s)$.
- Triển khai thuật toán **On-policy MC Control** với chiến lược $\epsilon$-greedy để tối ưu hóa bảng giá trị hành động $Q(s,a)$ và tìm chính sách tối ưu.
- Thực hiện mở rộng bài toán với thuật toán **Off-policy MC Control** áp dụng kỹ thuật **Weighted Importance Sampling**.
- Đánh giá, trực quan hóa đường cong huấn luyện (Learning Curve) và so sánh hiệu năng giữa các chính sách.

## Cài đặt

Cài đặt các thư viện cần thiết trước khi chạy chương trình:
- `pip install gymnasium numpy matplotlib`

## Cách chạy

1. **Chạy chương trình chính (Monte Carlo Agent hoàn chỉnh):**
   - `python src/main.py`
2. **Chạy bài tập mở rộng (Off-policy Monte Carlo Control):**
   - `python src/off_policy_mc.py`

## Episode và Return

- **Episode:** Một chuỗi đầy đủ các tương tác từ trạng thái bắt đầu đến khi kết thúc (Terminal state):
  $$S_0, A_0, R_1, S_1, A_1, R_2, \\dots, S_{T-1}, A_{T-1}, R_T$$
- **Return ($G_t$):** Tổng phần thưởng nhận được tính từ bước $t$ đến cuối episode, có sử dụng hệ số chiết khấu $\\gamma \\in [0, 1]$:
  $$G_t = \\sum_{k=0}^{T-t-1} \\gamma^k R_{t+k+1} = R_{t+1} + \\gamma G_{t+1}$$

## First-Visit MC

Trong phương pháp **First-Visit MC Prediction**, giá trị trạng thái $V(s)$ được tính bằng trung bình phần thưởng tích lũy ($G_t$) thu được từ lần xuất hiện **đầu tiên** của trạng thái $s$ trong mỗi episode:

$$V(s) = \\frac{\\sum \\text{Return thu được từ lần đầu ghé thăm } s}{\\text{Số episode chứa } s}$$

## Every-Visit MC

Trong phương pháp **Every-Visit MC Prediction**, giá trị $V(s)$ được cập nhật tích lũy cho **tất cả các lần** trạng thái $s$ xuất hiện trong từng episode (ngay cả khi trạng thái đó lặp lại nhiều lần trong cùng một episode):

$$V(s) = \\frac{\\sum \\text{Return thu được từ mọi lần ghé thăm } s}{\\text{Tổng số lần ghé thăm } s}$$

## Action-value Q(s,a)

Hàm giá trị hành động $Q(s,a)$ đại diện cho phần thưởng kỳ vọng khi thực hiện hành động $a$ tại trạng thái $s$ và tuân theo chính sách $\\pi$ sau đó. $Q(s,a)$ được cập nhật trung bình tăng dần (incremental update):

$$Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\frac{1}{N(S_t, A_t)} \\left[ G_t - Q(S_t, A_t) \\right]$$

## Epsilon-greedy

Chiến lược $\\epsilon$-greedy cân bằng giữa **Thám hiểm (Exploration)** và **Khai thác (Exploitation)**:
- Với xác suất $1 - \\epsilon$: Chọn hành động tốt nhất hiện tại (Greedy action): $\\arg\\max_a Q(s,a)$.
- Với xác suất $\\epsilon$: Chọn ngẫu nhiên một hành động trong không gian $A(s)$.

## On-policy MC Control

Thuật toán On-policy MC Control liên tục cải thiện chính sách bằng cách:
1. Thu thập dữ liệu episode thông qua chiến lược $\\epsilon$-greedy dựa trên $Q(s,a)$ hiện tại.
2. Cập nhật bảng giá trị $Q(s,a)$ bằng phương pháp First-Visit MC.
3. Cập nhật lại chính sách $\\epsilon$-greedy theo bảng $Q(s,a)$ mới thu được.

## Kết quả

Dưới đây là bảng đánh giá hiệu năng thu được sau khi kiểm thử 10,000 episodes:

| Policy | Win Rate | Loss Rate | Draw Rate | Mean Reward |
| :--- | :---: | :---: | :---: | :---: |
| **Random Policy** | ~28.2% | ~62.3% | ~9.5% | -0.3410 |
| **Fixed Policy (Stick on 20)** | ~40.2% | ~53.8% | ~6.0% | -0.1360 |
| **Learned On-policy MC** | ~42.5% | ~48.2% | ~9.3% | -0.0570 |
| **Learned Off-policy MC** | ~43.1% | ~47.8% | ~9.1% | -0.0470 |

## Learning Curve

Đường cong huấn luyện của Monte Carlo Control cho thấy phần thưởng trung bình di động (Moving Average Reward) tăng dần và ổn định theo thời gian qua $50,000$ episodes.

- **On-policy MC Control:** Ảnh lưu tại `Lab03/figures/bai36_mc_agent_training.png`
- **Off-policy MC Control:** Ảnh lưu tại `Lab03/figures/off_policy_mc_control.png`

## So sánh policy

1. **Random Policy:** Cho kết quả kém nhất do liên tục rút bài ngay cả khi tổng điểm đã cao, dẫn đến tỷ lệ đền bài (bust) rất lớn.
2. **Fixed Policy (Stick-on-20):** Cải thiện rõ rệt so với Random Policy nhưng chiến lược quá cứng nhắc, không tính tới lá bài ngửa của Dealer.
3. **Learned MC Policy (On-policy & Off-policy):** Đạt kết quả tối ưu nhất. Agent học được khi nào nên dừng hoặc rút dựa trên sự kết hợp giữa tổng điểm bản thân và bài của Dealer.

## Nhận xét

- Phương pháp Monte Carlo rất hiệu quả với các môi trường phân đoạn (episodic) như Blackjack mà không đòi hỏi phải biết mô hình xác suất chuyển trạng thái của môi trường $P(s'\vert{}s,a)$.
- Cả 2 phương pháp **First-Visit** và **Every-Visit** MC Prediction cho kết quả giá trị xấp xỉ tương đương nhau trên môi trường Blackjack do tính chất ít bị lặp lại trạng thái trong cùng một episode.
- Kỹ thuật **Off-policy MC** cho phép học một chính sách mục tiêu Greedy tối ưu hoàn toàn thông qua dữ liệu thu thập bởi chính sách thám hiểm Behavior Policy, tuy nhiên cần chú ý đến vấn đề biến động trọng số khi áp dụng Importance Sampling.

## Tài liệu tham khảo

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Farama Foundation. (n.d.). *Gymnasium Documentation - Blackjack-v1*.
