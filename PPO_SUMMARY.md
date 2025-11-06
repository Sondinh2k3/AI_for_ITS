# PPO Training - Tóm Tắt Nhanh

## 📦 Các File thuật toán PPO

```
scripts/
├── train_ppo.py              ← Script training PPO chính
├── eval_ppo.py               ← Script đánh giá model
├── ppo_config_examples.py    ← Ví dụ cấu hình (5 scenarios)
├── run_training.sh           ← Script menu tương tác
└── README_PPO.md             ← Hướng dẫn nhanh

TRAINING_GUIDE.md             ← Hướng dẫn chi tiết đầy đủ
```

## Bắt Đầu 

### 1. Setup (Lần đầu)

```bash
# Kích hoạt môi trường
source .venv/bin/activate

# Cài đặt dependencies
pip install -e .

# Kiểm tra
python -c "import ray; print('✓')"
```

### 2. Training Nhanh

```bash
# Test nhanh (10 iterations)
python scripts/train_ppo.py --iterations 10 --workers 1

# Training tiêu chuẩn (100 iterations)
python scripts/train_ppo.py --network grid4x4 --iterations 100 --workers 2

# Training cao cấp (500 iterations, GPU)
python scripts/train_ppo.py --network zurich --iterations 500 --workers 4 --gpu
```

### 3. Đánh Giá

```bash
# Sau training, copy đường dẫn checkpoint
CKPT="./results/ppo_grid4x4_.../checkpoint_000050"

# Đánh giá
python scripts/eval_ppo.py --checkpoint $CKPT --episodes 5
```

## Lõi Thuật toán PPO

**Công thức:**
```
L^CLIP(θ) = E_t [ min(r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t) ]
```

**Ưu điểm:**
- Ổn định, dễ tune
- Hiệu quả sample
- Hoạt động tốt cho traffic control

**Luồng:**
1. Rollout: Chạy environment với policy hiện tại
2. Compute: Tính advantages & returns (GAE)
3. Update: Multiple SGD passes trên batch
4. Repeat: Đến converge

## Tham Số Chính

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `learning_rate` | 5e-5 | Tốc độ học |
| `entropy_coeff` | 0.01 | Khuyến khích explore |
| `clip_param` | 0.3 | PPO clip range |
| `gamma` | 0.99 | Discount factor |
| `workers` | 2 | Parallel collection |

## Kết Quả

```
results/
└── ppo_grid4x4_20250104_120000/
    ├── checkpoint_000010/
    ├── checkpoint_000020/
    ├── training_config.json
    └── progress.csv
```

## Customize

Để thay đổi tham số, sửa trong `train_ppo.py`:

```python
def create_ppo_config(...):
    config = (
        PPOConfig()
        .training(
            lr=1e-4,              # ← Thay learning rate
            entropy_coeff=0.05,   # ← Thay entropy
            ...
        )
    )
```

Hoặc dùng examples từ `ppo_config_examples.py`:
- `get_ppo_config_small()` - Test nhanh
- `get_ppo_config_medium()` - Production (mặc định)
- `get_ppo_config_large()` - High performance
- `get_ppo_config_exploration()` - Khám phá cao
- `get_ppo_config_stable()` - Ổn định cao

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "SUMO_HOME not set" | `export SUMO_HOME=/usr/share/sumo` |
| Out of memory | Giảm workers: `--workers 1` |
| Training quá chậm | Tăng workers hoặc dùng GPU |
| Model không học | Tăng entropy_coeff hoặc learning_rate |

## Tài Liệu

- **Chi tiết:** `TRAINING_GUIDE.md`
- **Nhanh:** `scripts/README_PPO.md`
- **Ví dụ:** `scripts/ppo_config_examples.py`

---

**Không có thay đổi nào trong:**
- `src/environment/drl_algo/` (Environment)
- `pyproject.toml` (Dependencies)

**Các file mới:**
- ✨ `scripts/train_ppo.py` - Script training chính
- ✨ `scripts/eval_ppo.py` - Script evaluation
- ✨ `scripts/ppo_config_examples.py` - Config examples
- ✨ `scripts/run_training.sh` - Menu tương tác
- ✨ `TRAINING_GUIDE.md` - Hướng dẫn đầy đủ
