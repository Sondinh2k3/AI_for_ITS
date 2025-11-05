"""
Quick Start Summary for PPO Training

Các bước nhanh để bắt đầu huấn luyện mô hình PPO.
"""

# ============================================================================
# BƯỚC 1: CÀI ĐẶT
# ============================================================================

"""
1. Kích hoạt môi trường ảo:
   $ source .venv/bin/activate

2. Cài đặt dependencies:
   $ pip install -e .

3. Kiểm tra cài đặt:
   $ python -c "import ray; print('✓ Ready')"
"""

# ============================================================================
# BƯỚC 2: HUẤN LUYỆN
# ============================================================================

"""
CÁCH 1: Training nhanh (testing)
   $ python scripts/train_ppo.py --iterations 10 --workers 1

CÁCH 2: Training tiêu chuẩn (production)
   $ python scripts/train_ppo.py --network grid4x4 --iterations 100 --workers 2

CÁCH 3: Training cao cấp (tuning)
   $ python scripts/train_ppo.py \\
       --network zurich \\
       --iterations 500 \\
       --workers 4 \\
       --checkpoint-interval 20 \\
       --gpu

CÁCH 4: Dùng script menu tương tác
   $ bash scripts/run_training.sh
"""

# ============================================================================
# BƯỚC 3: ĐÁNH GIÁ
# ============================================================================

"""
Sau khi training xong:

   $ python scripts/eval_ppo.py \\
       --checkpoint ./results/ppo_grid4x4_20250104_120000/checkpoint_000050 \\
       --episodes 5
"""

# ============================================================================
# THAM SỐ CHỦ YẾU
# ============================================================================

PARAMETERS = {
    "network": {
        "default": "grid4x4",
        "options": ["grid4x4", "4x4loop", "zurich", "PhuQuoc"],
        "description": "Mạng SUMO để training",
    },
    "iterations": {
        "default": 100,
        "description": "Số lần update của agent",
        "recommendations": {
            "quick_test": 10,
            "medium": 100,
            "production": 500,
            "large": 1000,
        },
    },
    "workers": {
        "default": 2,
        "description": "Số worker để collect data song song",
        "recommendations": {
            "cpu_cores": "Nên = số CPU cores của máy",
            "memory": "Mỗi worker tốn ~200-500MB",
            "max_recommended": 8,
        },
    },
    "checkpoint_interval": {
        "default": 10,
        "description": "Lưu checkpoint mỗi N iterations",
    },
    "reward_threshold": {
        "default": None,
        "description": "Dừng training khi đạt reward này",
    },
    "gpu": {
        "default": False,
        "description": "Sử dụng GPU cho training",
        "note": "Cần CUDA, thường nhanh hơn 5-10x",
    },
    "gui": {
        "default": False,
        "description": "Hiển thị SUMO GUI",
        "note": "Làm chậm training, dùng khi debug",
    },
}

# ============================================================================
# FILE CẤU TRÚC
# ============================================================================

"""
scripts/
├── train_ppo.py              ← Script chính training PPO
├── eval_ppo.py               ← Script đánh giá model
├── ppo_config_examples.py    ← Ví dụ cấu hình PPO
├── run_training.sh           ← Script menu tương tác
└── train_rllib.py            (cũ - DQN)

src/environment/drl_algo/
├── env.py                    ← Environment SUMO (không thay đổi)
├── traffic_signal.py         ← Traffic signal logic (không thay đổi)
├── observations.py           ← Observation functions (không thay đổi)
└── resco_envs.py            ← RESCO envs (không thay đổi)

results/
└── ppo_grid4x4_20250104_120000/
    ├── checkpoint_000010/
    ├── checkpoint_000020/
    ├── ...
    └── training_config.json

TRAINING_GUIDE.md             ← Hướng dẫn chi tiết
README_PPO.md                 ← File này
"""

# ============================================================================
# WORKFLOW THƯỜNG NGÀY
# ============================================================================

"""
1. DEVELOPMENT (Debugging)
   $ python scripts/train_ppo.py --iterations 10 --workers 1
   → Nhanh, xem có lỗi gì không

2. TESTING (Validation)
   $ python scripts/train_ppo.py --network grid4x4 --iterations 50 --workers 2
   → Kiểm tra xem model học được không

3. TRAINING (Production)
   $ nohup python scripts/train_ppo.py --network zurich --iterations 500 --workers 4 &
   → Chạy background, có thể đóng terminal

4. EVALUATION (Assessment)
   $ python scripts/eval_ppo.py --checkpoint results/.../checkpoint_000100 --episodes 10
   → Kiểm tra performance trên test set

5. DEPLOYMENT (Sử dụng)
   → Load checkpoint trong ứng dụng thực tế
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
❌ Error: "SUMO_HOME is not set"
✅ Solution: export SUMO_HOME=/usr/share/sumo

❌ Error: "Ray not initialized"
✅ Solution: ray.init() được gọi tự động, nếu lỗi thì check Ray installation

❌ Error: "Out of memory"
✅ Solution: Giảm workers (--workers 1) hoặc batch size

❌ Model không học
✅ Solutions:
   - Kiểm tra environment có reward không
   - Tăng entropy_coeff (--entropy 0.05 trong code)
   - Tăng learning_rate (--lr 1e-4 trong code)
   - Kiểm tra network architecture

❌ Training quá chậm
✅ Solutions:
   - Tăng workers
   - Sử dụng GPU (--gpu)
   - Giảm network size
"""

# ============================================================================
# MONITORING TRAINING
# ============================================================================

"""
Theo dõi tiến độ:

1. Xem output trong terminal:
   Iteration   1 | Episode Reward Mean:   -12.45 | Episode Len Mean:   250.0
   Iteration   2 | Episode Reward Mean:   -10.32 | Episode Len Mean:   255.3
   ...

2. Kiểm tra file training_config.json:
   $ cat results/ppo_*/training_config.json

3. Plot results (nếu có):
   $ python -c "import pandas as pd; df=pd.read_csv('results/.../progress.csv'); df.plot()"

4. Xem checkpoint được tạo:
   $ ls -lh results/ppo_grid4x4_*/checkpoint_*
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
Sau khi training và eval thành công:

1. Tuning hyperparameters:
   - Xem ppo_config_examples.py
   - Thử learning_rate khác
   - Thử entropy_coeff khác

2. Đánh giá trên nhiều episodes:
   - eval_ppo.py --episodes 50

3. So sánh models:
   - Lưu kết quả từ eval vào file
   - So sánh performance giữa các checkpoint

4. Deploy:
   - Load checkpoint trong ứng dụng
   - Tích hợp vào SUMO backend
   - Monitor performance real-world
"""

# ============================================================================
# REFERENCES
# ============================================================================

"""
📚 Papers:
   - PPO: https://arxiv.org/abs/1707.06347
   - Trust Region Policy Optimization: https://arxiv.org/abs/1502.05477

📖 Documentation:
   - Ray RLlib: https://docs.ray.io/en/latest/rllib/
   - Gymnasium: https://gymnasium.farama.org/
   - SUMO: https://sumo.dlr.de/

💡 Tips:
   - Start small and scale up
   - Monitor reward closely
   - Save best models
   - Document experiments
"""

if __name__ == "__main__":
    print("Xem TRAINING_GUIDE.md để có hướng dẫn chi tiết!")
