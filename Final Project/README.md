# Coin Collector với đồng xu biến mất

So sánh các thuật toán học tăng cường dạng bảng trên bài toán thu thập đồng xu có thời gian giới hạn:
Q-Learning, Monte Carlo, Sarsa (kèm biến thể khởi tạo lạc quan) và các thuật toán dựa trên mô hình
Value Iteration, Policy Iteration, Truncated Policy Iteration. Nghiệm tối ưu tính bằng quy hoạch động
được dùng làm mốc đối chiếu.

## Bài toán

Robot di chuyển trên lưới, nhặt đồng xu rồi tới ô Exit trước khi hết bước.

- **Hành động:** Lên, Xuống, Trái, Phải. Đâm tường hoặc biên thì đứng yên nhưng vẫn mất một bước.
- **Trạng thái:** `(hàng, cột, coin_mask, time_remaining, bonus_used)`, đủ thông tin để bảo đảm tính Markov.
- **Đồng xu biến mất:** một số đồng xu biến mất sau một số bước nhất định.
- **Ô Time Bonus:** cộng thêm bước đi, chỉ dùng được một lần.
- **Môi trường tất định:** lớp `CoinCollectorEnv` kế thừa `gymnasium.Env`.
- **Điểm nhiệm vụ:** tổng giá trị đồng xu thu được + 10 nếu tới Exit. Chỉ số này không phụ thuộc hàm phần thưởng nên dùng để so sánh chung.
- **5 hàm phần thưởng:** R1 (thưa), R2 (theo khoảng cách), R3 (khẩn cấp), R4 (theo thời gian), R5 (lai ghép).
- **4 bản đồ:** Map 1 (4x4), Map 2 (6x6), Map 3 (8x8), Map 4 (7x7, có ngõ cụt và Time Bonus +8).

## Yêu cầu

- Python 3.13 (notebook được chạy với 3.13.14)
- Các thư viện trong `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách chạy

```bash
jupyter notebook Coin_Collector.ipynb
```

Chọn **Kernel -> Restart & Run All**. Các cell phải chạy theo thứ tự từ trên xuống vì các phần sau dùng kết quả của phần trước (`RES`, `HP`, `DP`).

**Chạy nhanh để kiểm tra mã:** đặt biến môi trường `CC_QUICK=1` trước khi mở notebook (2 seed, số episode giảm còn 4%).

```bash
CC_QUICK=1 jupyter notebook Coin_Collector.ipynb     # Windows PowerShell: $env:CC_QUICK="1"; jupyter notebook
```

## Cấu trúc notebook

| Phần | Nội dung |
|---|---|
| 1 | Thiết lập: seed, số episode, `GAMMA = 0.99`, lịch giảm epsilon |
| 2 | Mô hình MDP: agent, trạng thái, hành động, return, 5 hàm phần thưởng |
| 3 | Môi trường `CoinCollectorEnv`, 4 bản đồ, nghiệm tối ưu bằng quy hoạch động, kích thước không gian trạng thái |
| 4 | Thuật toán: Q-Learning, Monte Carlo, khởi tạo lạc quan, Sarsa, VI, PI, TPI; bộ chạy song song |
| 5 | Tinh chỉnh siêu tham số (Map 2 và 3, R3, 3 seed riêng) |
| 6 | Thí nghiệm chính: 4 bản đồ x 5 reward x 4 thuật toán x 5 seed = 400 lượt; bảng, heatmap, đường học, chính sách |
| 7 | Phân tích nguyên nhân: loại trừ lỗi reward, quyết định sai đầu tiên, sai số Q, can thiệp I1 đến I6 |
| 8 | Độ nhạy theo môi trường: ngân sách bước, hạn biến mất, Time Bonus |
| 9 | Sarsa, VI, PI, TPI và so sánh thống nhất chất lượng/chi phí |
| 10 | Demo một episode bằng chính sách đã học |

## Đầu ra

- Thư mục `configs/` với `map1.json` đến `map4.json` được tạo tự động ở Phần 3.
- Bảng và biểu đồ hiển thị trực tiếp trong notebook; tiêu đề và nhãn biểu đồ bằng tiếng Anh.
- Kết quả tái lập được: seed cố định `[42, 123, 456, 789, 2024]` cho thí nghiệm chính và `[7, 17, 27]` cho tinh chỉnh.

## Kết quả chính

| Thuật toán | Tỷ lệ so với tối ưu (trung bình 4 bản đồ) |
|---|---|
| QL-opt | 1,000 (đạt tối ưu ở 100/100 lượt) |
| Sarsa-opt | 0,980 |
| MC | 0,867 |
| MC-opt | 0,858 |
| Sarsa | 0,852 |
| QL | 0,802 |
| VI, PI, TPI | 1,000 (nghiệm chính xác, cần biết mô hình) |

Chính sách tối ưu theo cả năm hàm phần thưởng đều đạt điểm nhiệm vụ tối ưu, nên không lượt chạy nào thất bại do hàm phần thưởng.
Phần lớn thất bại của Q-Learning không khởi tạo lạc quan là do khám phá sâu.
