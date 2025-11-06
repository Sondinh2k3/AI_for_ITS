# PPO Training for Adaptive Traffic Signal Control

Hướng dẫn chi tiết về huấn luyện mô hình PPO (Proximal Policy Optimization) để điều khiển đèn giao thông thích ứng trong SUMO.

## Mục Lục

1. [Cấu trúc File](#cấu-trúc-file)
2. [Cài đặt Môi trường](#cài-đặt-môi-trường)
3. [Huấn luyện Mô hình](#huấn-luyện-mô-hình)
4. [Đánh giá Mô hình](#đánh-giá-mô-hình)
5. [Tham số Cấu hình](#tham-số-cấu-hình)
6. [Cấu trúc PPO](#cấu-trúc-ppo)
7. [Kết quả Huấn luyện](#kết-quả-huấn-luyện)

---

## Cấu trúc File

```
AI_for_ITS/
├── scripts/
│   ├── train_ppo.py          # Script huấn luyện PPO ← TẠO MỚI
│   ├── eval_ppo.py           # Script đánh giá mô hình ← TẠO MỚI
│   └── train_rllib.py        # Script huấn luyện DQN (cũ)
├── src/environment/drl_algo/
│   ├── env.py                # SumoEnvironment (không thay đổi)
│   ├── traffic_signal.py     # TrafficSignal class (không thay đổi)
│   ├── observations.py       # Observation functions (không thay đổi)
│   └── resco_envs.py         # RESCO environments (không thay đổi)
├── network/
│   ├── grid4x4/              # Grid 4x4 network
│   ├── 4x4loop/              # 4x4 Loop network
│   ├── zurich/               # Zurich network
│   └── PhuQuoc/              # Phu Quoc network
├── pyproject.toml            # Dependencies (không thay đổi)
└── results/                  # Thư mục lưu kết quả (tạo tự động)
    └── ppo_grid4x4_20250104_120000/
        ├── checkpoint_000010/
        ├── checkpoint_000020/
        └── training_config.json
```

---

## Cài đặt Môi trường

### 1. Tạo và Kích hoạt Môi trường Ảo

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường (Linux/macOS)
source .venv/bin/activate

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
```

### 2. Cài đặt Dependencies

```bash
# Nâng cấp pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Cài đặt các dependencies từ pyproject.toml
pip install -e .

# Hoặc cài từng package
pip install "pettingzoo>=1.25.0"
pip install "gymnasium>=1.1.1"
pip install "ray[rllib]>=2.50.1"
pip install "numpy>=1.24.0"
pip install "pyvirtualdisplay>=3.0"
```

### 3. Kiểm tra Cài đặt

```bash
python -c "import ray; import gymnasium; print('✓ Installation successful')"
```

### 4. Đặt Biến Môi trường SUMO_HOME

```bash
# Linux/macOS
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH

# Hoặc thêm vào ~/.bashrc hoặc ~/.zshrc
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
echo 'export PATH=$SUMO_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Huấn luyện Mô hình

### Cách 1: Sử dụng Lệnh Đơn giản

```bash
# Huấn luyện với cài đặt mặc định (grid4x4, 100 iterations)
python scripts/train_ppo.py

# Huấn luyện network khác
python scripts/train_ppo.py --network zurich

# Huấn luyện 500 iterations với 4 workers
python scripts/train_ppo.py --network grid4x4 --iterations 500 --workers 4

# Huấn luyện với GPU
python scripts/train_ppo.py --gpu

# Huấn luyện với SUMO GUI
python scripts/train_ppo.py --gui
```

### Cách 2: Xem Đầy đủ các Tham số

```bash
python scripts/train_ppo.py --help
```

**Output:**
```
usage: train_ppo.py [-h] [--network {grid4x4,4x4loop,network_test,zurich,PhuQuoc}]
                     [--iterations ITERATIONS]
                     [--workers WORKERS]
                     [--checkpoint-interval CHECKPOINT_INTERVAL]
                     [--reward-threshold REWARD_THRESHOLD]
                     [--experiment-name EXPERIMENT_NAME]
                     [--gui]
                     [--gpu]
                     [--seed SEED]
                     [--output-dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --network {grid4x4,4x4loop,network_test,zurich,PhuQuoc}
                        Network name (default: grid4x4)
  --iterations ITERATIONS
                        Number of training iterations (default: 100)
  --workers WORKERS     Number of parallel workers (default: 2)
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Checkpoint interval (default: 10)
  --reward-threshold REWARD_THRESHOLD
                        Stop training if reward exceeds this threshold
  --experiment-name EXPERIMENT_NAME
                        Experiment name for logging
  --gui                 Use SUMO GUI
  --gpu                 Use GPU for training
  --seed SEED           Random seed (default: 42)
  --output-dir OUTPUT_DIR
                        Output directory for results (default: ./results)
```

### Cách 3: Ví dụ Thực tế

```bash
# Huấn luyện thử nghiệm nhanh (10 iterations, 1 worker)
python scripts/train_ppo.py \
  --network grid4x4 \
  --iterations 10 \
  --workers 1 \
  --checkpoint-interval 5 \
  --seed 123

# Huấn luyện sản xuất (500 iterations, 4 workers, GPU)
python scripts/train_ppo.py \
  --network zurich \
  --iterations 500 \
  --workers 4 \
  --checkpoint-interval 20 \
  --gpu \
  --seed 42 \
  --experiment-name "ppo_zurich_final"

# Huấn luyện với ngưỡng dừng (stop khi reward > 100)
python scripts/train_ppo.py \
  --network 4x4loop \
  --iterations 1000 \
  --reward-threshold 100 \
  --workers 4
```

---

## Đánh giá Mô hình

### Sau Huấn luyện

```bash
# Lấy đường dẫn checkpoint từ results
CHECKPOINT_PATH="./results/ppo_grid4x4_20250104_120000/checkpoint_000050"

# Đánh giá 5 episodes
python scripts/eval_ppo.py --checkpoint $CHECKPOINT_PATH

# Đánh giá 10 episodes trên network khác
python scripts/eval_ppo.py \
  --checkpoint $CHECKPOINT_PATH \
  --network zurich \
  --episodes 10

# Đánh giá với SUMO GUI
python scripts/eval_ppo.py \
  --checkpoint $CHECKPOINT_PATH \
  --gui \
  --episodes 3
```

---

## Tham số Cấu hình

### Tham số SUMO Environment

| Tham số | Giá trị Mặc định | Mô tả |
|---------|-----------------|-------|
| `max_green` | 60s | Thời gian xanh tối đa |
| `min_green` | 5s | Thời gian xanh tối thiểu |
| `delta_time` | 5s | Khoảng thời gian giữa các action |
| `yellow_time` | 3s | Thời gian vàng |
| `use_gui` | False | Sử dụng SUMO GUI |

### Tham số PPO Training

| Tham số | Giá trị Mặc định | Mô tả |
|---------|-----------------|-------|
| `learning_rate` | 5e-5 | Tốc độ học |
| `gamma` | 0.99 | Discount factor |
| `lambda_` | 0.95 | GAE lambda |
| `entropy_coeff` | 0.01 | Hệ số entropy (exploration) |
| `clip_param` | 0.3 | PPO clip parameter |
| `fcnet_hiddens` | [256, 256] | Kích thước hidden layers |
| `sgd_minibatch_size` | 128 | Batch size cho SGD |
| `train_batch_size` | 4096 | Tổng batch size cho training |
| `num_sgd_iter` | 30 | Số lần update SGD mỗi iteration |

### Điều chỉnh Tham số

Để thay đổi tham số PPO, chỉnh sửa hàm `create_ppo_config()` trong `train_ppo.py`:

```python
def create_ppo_config(...):
    config = (
        PPOConfig()
        ...
        .training(
            lr=1e-4,              # ← Tăng learning rate
            entropy_coeff=0.02,   # ← Tăng exploration
            clip_param=0.2,       # ← Giảm clip param (conservative)
            ...
        )
    )
```

---

## Cấu trúc PPO

### Thuật toán PPO (Proximal Policy Optimization)

**Ưu điểm:**
- Ổn định hơn policy gradient methods
- Hiệu quả sample-efficient
- Dễ implement và debug
- Hoạt động tốt với both discrete và continuous actions

**Công thức Update:**

```
L^CLIP(θ) = E_t [ min(r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t) ]

Trong đó:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (Probability Ratio)
  Â_t = Q(s_t, a_t) - V(s_t)               (Advantage Estimate)
  ε = clip_param = 0.3                      (Clip range)
```

**Luồng Huấn luyện:**

```
1. Rollout phase:
   - Chạy environment với current policy
   - Lấy {s_t, a_t, r_t, s_{t+1}}
   - Tính advantages bằng GAE

2. Update phase (multiple epochs):
   - Shuffle trajectories
   - Mini-batch updates
   - Calculate PPO loss
   - Backward pass
   - Update parameters

3. Repeat đến khi converge hoặc max iterations
```

### Network Architecture

```
State Input (Observation)
    ↓
Linear Layer (256, relu)
    ↓
Linear Layer (256, relu)
    ↓
┌─────────────────┬──────────────────┐
↓                 ↓
Policy Head   Value Head
(Actor)       (Critic)
    ↓             ↓
Output: πθ    Output: V(s)
(Action)      (State Value)
```

---

## Kết quả Huấn luyện

### Cấu trúc Thư mục Kết quả

```
results/
└── ppo_grid4x4_20250104_120000/
    ├── checkpoint_000010/
    │   ├── algorithm_state.pkl
    │   ├── policy_0
    │   │   ├── model.pkl
    │   │   └── rllib_checkpoint.json
    │   ├── rllib_checkpoint.json
    │   └── training_iteration
    ├── checkpoint_000020/
    ├── checkpoint_000030/
    ├── training_config.json
    └── progress.csv
```

### File Log

**training_config.json**: Lưu cấu hình training
```json
{
  "experiment_name": "ppo_grid4x4_20250104_120000",
  "network_name": "grid4x4",
  "num_iterations": 100,
  "num_workers": 2,
  "checkpoint_interval": 10,
  "use_gpu": false,
  "seed": 42,
  "best_checkpoint": "./results/ppo_grid4x4_20250104_120000/checkpoint_000050",
  "best_reward": 125.45
}
```

### Theo dõi Tiến độ

Khi training, bạn sẽ thấy output như:

```
Iteration   1 | Episode Reward Mean:   -12.45 | Episode Len Mean:   250.0
Iteration   2 | Episode Reward Mean:   -10.32 | Episode Len Mean:   255.3
Iteration   3 | Episode Reward Mean:     5.67 | Episode Len Mean:   265.1
Iteration   4 | Episode Reward Mean:    18.90 | Episode Len Mean:   268.5
...
Iteration 100 | Episode Reward Mean:   125.45 | Episode Len Mean:   285.2

✓ Training completed: reached max iterations (100)
```

---

## Troubleshooting

### Lỗi: "SUMO_HOME not set"

```bash
# Kiểm tra SUMO_HOME
echo $SUMO_HOME

# Nếu rỗng, đặt biến môi trường
export SUMO_HOME=/usr/share/sumo
```

### Lỗi: "Out of Memory"

```bash
# Giảm số workers
python scripts/train_ppo.py --workers 1

# Hoặc giảm batch size trong code
# (sửa sgd_minibatch_size, train_batch_size)
```

### Training Chậm

```bash
# Tăng số workers (nếu đủ CPU)
python scripts/train_ppo.py --workers 8

# Hoặc sử dụng GPU
python scripts/train_ppo.py --gpu
```

### Mô hình Không Học

- ✓ Kiểm tra environment có reward không
- ✓ Tăng learning_rate hoặc entropy_coeff
- ✓ Giảm clip_param (mặc định 0.3)
- ✓ Kiểm tra lại initialization

---

## Tài liệu Tham khảo

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Ray RLlib Docs](https://docs.ray.io/en/latest/rllib/index.html)
- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Gymnasium Docs](https://gymnasium.farama.org/)

---

## Ghi Chú

- Script này không thay đổi code trong `src/environment/drl_algo/`
- Mô hình được lưu dưới dạng checkpoint có thể load lại để tiếp tục training
- Mỗi lần training được tạo một thư mục riêng với timestamp để tránh ghi đè
- Bạn có thể chạy nhiều training song song trên các worker khác nhau

---
