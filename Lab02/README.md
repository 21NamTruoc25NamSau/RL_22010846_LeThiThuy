# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên

- Họ tên: Lê Thị Thuý
- MSSV: 22010846
- Lớp: Kỹ Thuật Robot Và Trí Tuệ Nhân Tạo (K16)
- Học phần: Học Tăng Cường Cơ Bản (N01)
- GitHub username: 21namtruoc25namsau

## Mục tiêu

- Hiểu và cài đặt Markov chain, tính transition probability và mô phỏng bằng sampling.
- Hiểu khái niệm reward, return, discount factor và ảnh hưởng của gamma.
- Xây dựng và kiểm tra một Markov Decision Process (MDP) tự thiết kế.
- Khám phá transition model thật của môi trường `FrozenLake-v1` (Gymnasium).
- Cài đặt từ đầu (không dùng thư viện RL có sẵn) các thuật toán Dynamic Programming kinh điển: Bellman backup, Policy Evaluation, Policy Iteration, Value Iteration.
- Đánh giá và so sánh các policy/thuật toán bằng cả lý thuyết lẫn mô phỏng thực nghiệm (simulation).
- Tổng hợp toàn bộ kiến thức thành một chương trình Dynamic Programming Solver hoàn chỉnh (`main.py`).

## Cấu trúc thư mục

Lab02/
├── README.md
├── requirements.txt
├── src/
│ ├── bai01.py
│ ├── bai02.py
│ ├── ...
│ ├── bai36.py
│ ├── mdp_utils.py
│ └── main.py
├── notebooks/
│ └── Lab02_22010846_LeThiThuy.ipynb
├── figures/
│ ├── markov_distribution.png
│ ├── gamma_comparison.png
│ ├── value_iteration_convergence.png
│ ├── policy_iteration_convergence.png
│ ├── algorithm_comparison.png
│ └── dp_solver_convergence.png
└── data/
  └── README.md


## Cài đặt

Yêu cầu Python ≥ 3.10. Cài đặt các thư viện cần thiết:

```bash
cd Lab02
pip install -r requirements.txt
```

## Cách chạy

```bash
cd Lab02
pip install -r requirements.txt
python src/bai01.py
python src/bai24.py
python src/bai29.py
python src/bai32.py
python src/main.py
```

Chạy notebook tổng hợp (bao gồm toàn bộ 36 bài, kết quả, biểu đồ và trả lời câu hỏi lý thuyết):

```bash
jupyter notebook
```

sau đó mở file `notebooks/Lab02_22010846_LeThiThuy.ipynb` và chạy tuần tự từ trên xuống (Cell → Run All), lưu ý cell **Setup** và cell tạo `mdp_utils.py` phải được chạy trước các bài từ Bài 21 trở đi.

## Thuật toán đã cài đặt

### Policy Evaluation

Tính giá trị `V(s)` của một policy π cố định bằng cách lặp cập nhật Bellman expectation equation: 
V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γV(s')]

cho đến khi `max|V_new - V_old| < θ`. Cài đặt tại `policy_evaluation()` / `policy_evaluation_sweep()` trong `mdp_utils.py` (Bài 23–25).

### Policy Iteration

Lặp lại hai bước: (1) Policy Evaluation để tính `V(s)` của policy hiện tại, (2) Policy Improvement để tạo policy mới bằng cách chọn hành động tối ưu (`argmax_a Q(s,a)`) tại mỗi state. Dừng khi policy không còn thay đổi (policy stable). Cài đặt tại `policy_iteration()` trong `mdp_utils.py` (Bài 26–30).

### Value Iteration

Kết hợp evaluation và improvement trong cùng một bước cập nhật, dùng Bellman optimality equation:
V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]

lặp tới khi hội tụ (`delta < θ`), sau đó trích xuất policy tối ưu bằng một bước greedy cuối cùng. Cài đặt tại `value_iteration()` trong `mdp_utils.py` (Bài 31–33).

## Kết quả FrozenLake

Trên `FrozenLake-v1` (4x4, `is_slippery=True`, gamma=0.99, theta=1e-8):

- Optimal policy (dạng lưới, `H`=Hole, `G`=Goal):
← ↑ ↑ ↑
← H ← H
↑ ↓ ← H
H → ↓ G

- Success rate khi đánh giá bằng 1000 episode mô phỏng: **72.1%** (so với chỉ 3.3% của random policy).
- Mở rộng trên bản đồ 8x8: success rate giảm còn 65.3% do đường đi dài hơn và nhiều hố băng hơn (xem Mục 10 – Mở rộng 1).

## So sánh Value Iteration và Policy Iteration

| Thuật toán       | Số vòng lặp | Thời gian chạy | Success rate | Mean reward |
|------------------|:-----------:|:--------------:|:------------:|:-----------:|
| Value Iteration  | 438         | ≈0.055–0.075s   | 0.721        | 0.721       |
| Policy Iteration | 3           | ≈0.040–0.049s   | 0.721        | 0.721       |

Cả hai thuật toán hội tụ về cùng một optimal policy và optimal value function. Policy Iteration cần rất ít vòng lặp ngoài vì mỗi vòng đã bao gồm một Policy Evaluation đầy đủ (nhưng nội bộ vòng đó lại tốn nhiều sweep); Value Iteration cần nhiều iteration hơn vì mỗi sweep chỉ cải thiện một phần bằng cách lấy `max` thay vì chờ hội tụ hoàn toàn. Trên bài toán nhỏ như FrozenLake 4x4, runtime thực tế của hai thuật toán khá tương đương.

## Nhận xét

- Cả Markov chain, MDP tự thiết kế, và `FrozenLake-v1` đều minh họa rõ vai trò trung tâm của transition model trong Dynamic Programming: mọi phép cập nhật Bellman đều cần duyệt đầy đủ danh sách `(probability, next_state, reward, terminated)`.
- Discount factor (gamma) ảnh hưởng trực tiếp đến cả tốc độ hội tụ (gamma càng lớn, số iteration cần thiết càng nhiều) và chất lượng policy tìm được (gamma quá nhỏ khiến agent "недалеко nhìn", bỏ lỡ phần thưởng ở xa).
- Value Iteration và Policy Iteration là hai cách tiếp cận khác nhau để giải cùng một bài toán tối ưu Bellman, và trên các bài toán nhỏ chúng cho kết quả giống hệt nhau; sự khác biệt chủ yếu nằm ở cấu trúc vòng lặp và chi phí tính toán từng bước.
- Khi mở rộng sang bản đồ lớn hơn (8x8), chi phí tính toán tăng nhanh hơn số state theo cấp số nhân nhẹ, cho thấy hạn chế "curse of dimensionality" của Dynamic Programming dạng bảng (tabular) — đây chính là động lực cho các phương pháp model-free / hàm xấp xỉ (function approximation) ở các lab sau.
- Việc đánh giá bằng simulation (Bài 34) là bước kiểm chứng bắt buộc: một optimal policy đúng trên lý thuyết vẫn có thể sai trong một số episode cụ thể do bản chất ngẫu nhiên của `is_slippery=True`, nên success rate mô phỏng phản ánh chân thực hơn hiệu quả thực tế của policy.

