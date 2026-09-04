# Bài thực hành số 1: Làm quen với môi trường Học Tăng Cường và Gymnasium

**Học phần:** Học Tăng Cường Cơ Bản (Reinforcement Learning) – N01

**Sinh viên thực hiện:** Lê Thị Thúy

**Mã sinh viên:** 22010846

**Lớp:** Kỹ thuật Robot và Trí Tuệ Nhân Tạo

## 1. Mục tiêu

- Cài đặt môi trường lập trình và thư viện Gymnasium.
- Hiểu vòng lặp tương tác `Agent → Action → Environment → Observation + Reward → Agent`.
- Phân biệt các khái niệm: environment, agent, observation/state, action, reward, episode, policy.
- Làm việc với `action_space`, `observation_space` và các API `reset()`, `step()` của Gymnasium.
- Xây dựng, đánh giá và so sánh các policy (random, heuristic) trên hai môi trường `CartPole-v1` (liên tục) và `FrozenLake-v1` (rời rạc).

## 2. Cấu trúc thư mục

```
Lab01/
├── README.md                       # File mô tả bài làm và hướng dẫn chạy
├── src/                            # 36 file Python (bai01.py -> bai35.py, main.py)
│   ├── bai01.py ... bai35.py       # Code từng bài tập, có thể chạy độc lập
│   ├── main.py                     # Mini-project: agent hoàn chỉnh (Bài 36)
│   └── migration_gym_to_gymnasium.py  # So sánh API Gym cũ và Gymnasium mới
├── figures/                        # Biểu đồ reward/moving average được lưu ra từ notebook
├── notebooks/
│   ├── Lab01_22010846_LeThiThuy.ipynb # Notebook chính chứa toàn bộ code, output, biểu đồ và nhận xét
└── data/                           # (dự phòng cho dữ liệu thực nghiệm)
│   ├── README.md
```

> Lưu ý: cây thư mục `src/`, `figures/`, `notebooks/`, `data/` cùng 36 file `bai*.py` được **sinh tự động** khi chạy các ô đầu tiên của notebook (notebook tự ghi file bằng lệnh `open(..., "w")`).

## 3. Yêu cầu môi trường

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.x |
| Gymnasium | 1.3.0 |
| NumPy | mới nhất tương thích |
| Matplotlib | mới nhất tương thích |

## 4. Hướng dẫn cài đặt

### 4.1. Tạo virtual environment (khuyến nghị)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 4.2. Cài đặt thư viện

```bash
python -m pip install --upgrade pip
pip install "gymnasium[classic-control,toy-text]==1.3.0"
pip install numpy matplotlib jupyter
```

### 4.3. Kiểm tra phiên bản

```bash
python --version
pip show gymnasium
```

## 5. Hướng dẫn chạy

### 5.1. Chạy toàn bộ notebook

Mở `Lab01_22010846_LeThiThuy.ipynb` bằng Jupyter Notebook/JupyterLab hoặc Google Colab, sau đó chạy tuần tự từ trên xuống (`Run All`). Notebook sẽ:

1. Cài đặt thư viện cần thiết.
2. Tạo cây thư mục `Lab01/src`, `figures/`, `notebooks/`, `data/`.
3. Sinh và thực thi lần lượt 36 file bài tập (`bai01.py` → `bai35.py`, `main.py`).
4. In kết quả, vẽ và lưu biểu đồ vào thư mục `figures/`.

```bash
jupyter notebook Lab01_22010846_LeThiThuy.ipynb
```

### 5.2. Chạy từng bài độc lập

Sau khi notebook đã sinh ra các file trong `src/`, mỗi bài có thể chạy riêng lẻ bằng dòng lệnh, ví dụ:

```bash
python src/bai01.py   # Kiểm tra môi trường Python & Gymnasium
python src/bai17.py   # Vẽ reward theo episode
python src/bai35.py   # So sánh 3 agent: Random / Angle-based / Improved
python src/main.py    # Mini-project agent hoàn chỉnh
```

## 6. Nội dung bài tập

| Phần | Chủ đề | Bài |
|---|---|---|
| A | Cài đặt và khám phá Gymnasium | 1–6 |
| B | Tương tác Agent – Environment (`step`, `reset`) | 7–12 |
| C | Episode và thống kê thực nghiệm (mean/std/max/min reward) | 13–18 |
| D | Random seed và khả năng tái lập | 19–22 |
| E | Môi trường rời rạc FrozenLake | 23–28 |
| F | Policy và cải thiện agent (heuristic) | 29–32 |
| G | Tổ chức code như một thí nghiệm RL hoàn chỉnh | 33–36 |
| Bắt buộc | Chuyển đổi code từ OpenAI Gym cũ sang Gymnasium | — |

## 7. Tóm tắt kết quả nổi bật

- **Random Policy** trên CartPole-v1: reward trung bình dao động quanh 20–23, độ lệch chuẩn lớn (~10), thể hiện tính ngẫu nhiên.
- **Angle-based Policy** (dựa trên `pole_angle`): cải thiện đáng kể, mean reward ~41, ổn định hơn Random.
- **Improved Policy** (kết hợp `pole_angle` và `pole_angular_velocity`): đạt reward tối đa 500.0 với độ lệch chuẩn 0.00 — giữ thăng bằng hoàn hảo trong suốt episode.
- **FrozenLake-v1** với random policy: success rate rất thấp (0.6%–3%), so sánh giữa chế độ deterministic (`is_slippery=False`) và stochastic (`is_slippery=True`) cho thấy sự khác biệt về độ ổn định và tỉ lệ thành công.
- Thiết lập `seed` cho `env.reset()` và `env.action_space.seed()` giúp đảm bảo khả năng tái lập kết quả thực nghiệm.

## 8. Ghi chú kỹ thuật (Gym cũ → Gymnasium)

- `env.reset()` trả về tuple `(observation, info)` thay vì chỉ `observation`.
- `env.step()` trả về 5 giá trị `(observation, reward, terminated, truncated, info)` thay vì 4 giá trị `(observation, reward, done, info)` như API cũ.
- Điều kiện kết thúc episode nên dùng `terminated or truncated` thay cho biến `done` cũ.
