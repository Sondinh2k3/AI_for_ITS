# Dự án Điều khiển Đèn giao thông Thích ứng sử dụng Học tăng cường (RL/DRL/MADRL)

Dự án này là một môi trường nghiên cứu và phát triển các thuật toán Học tăng cường (Reinforcement Learning - RL), Học tăng cường sâu (DRL) và Học tăng cường đa tác tử (MADRL) cho bài toán điều khiển đèn giao thông thích ứng. Mục tiêu là tối ưu hóa luồng giao thông, giảm thời gian chờ và giảm tắc nghẽn bằng cách sử dụng các trình mô phỏng như SUMO.

---

## Bắt đầu nhanh (Getting Started)

### 1. Yêu cầu hệ thống

* Python (khuyến nghị 3.9+)
* Trình mô phỏng (SUMO)
* Git

### 2. Cài đặt

1.  **Clone dự án:**
    ```bash
    git clone [URL_DU_AN]
    cd [TEN_DU_AN]
    ```

2.  **Cài đặt Poetry:**
    Dự án này sử dụng [Poetry](https://python-poetry.org/) để quản lý các thư viện. Nếu bạn chưa có Poetry, hãy cài đặt nó theo hướng dẫn trên trang chủ.

    *Trên macOS / Linux / WSL:*
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    *Trên Windows (sử dụng PowerShell):*
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

3.  **Cài đặt các thư viện của dự án:**
    Poetry sẽ đọc file `pyproject.toml`, tự động tạo một môi trường ảo và cài đặt tất cả các thư viện cần thiết.
    ```bash
    poetry install
    ```

4.  **Kích hoạt môi trường ảo:**
    Để kích hoạt môi trường ảo do Poetry quản lý, hãy chạy lệnh sau trong thư mục gốc của dự án:
    ```bash
    poetry shell
    ```
    Bây giờ bạn đã sẵn sàng để chạy các script của dự án.

5.  **Quản lý thư viện với Poetry:**
    Sử dụng các lệnh sau để quản lý các thư viện phụ thuộc của dự án:

    - **Thêm một thư viện mới:**
      ```bash
      poetry add <tên-thư-viện>
      ```
      *Ví dụ:* `poetry add gymnasium`

    - **Xóa một thư viện:**
      ```bash
      poetry remove <tên-thư-viện>
      ```

    - **Cập nhật các thư viện lên phiên bản mới nhất (theo ràng buộc trong `pyproject.toml`):**
      ```bash
      poetry update
      ```

---

## Cách sử dụng (Usage)

Các file thực thi chính nằm ở thư mục gốc của dự án.

### 1. Huấn luyện (Training)

Để bắt đầu một lượt huấn luyện mới, sử dụng `train.py` và chỉ định file cấu hình:

```bash
python train.py --config src/config/dqn_config.yaml
```

Tất cả các tệp mô hình (model checkpoints) sẽ được lưu vào results/models/.

Các tệp log (ví dụ: cho TensorBoard) sẽ được lưu vào results/logs/.

### 2. Đánh giá (Evaluation)

Để đánh giá một mô hình đã huấn luyện, sử dụng evaluate.py:

```Bash
python evaluate.py --model results/models/ten_model_da_huan_luyen.zip
```

## Cấu trúc Thư mục

Dưới đây là giải thích về cấu trúc thư mục của dự án:

.
├── docs/               #  Chứa tất cả tài liệu, ghi chú, báo cáo của dự án.
├── network/            #  Chứa các file định nghĩa mạng lưới giao thông (.sumocfg, .net.xml, .json...).
├── src/                #  Mã nguồn chính của dự án.
│   ├── algorithms/     #  Nơi chứa mã nguồn lõi của các thuật toán (DQN, PPO, MADDPG...).
│   ├── environment/    #  Định nghĩa các môi trường RL (Gym/PettingZoo) làm cầu nối với SUMO/CityFlow.
│   ├── config/         #  Các file cấu hình (.yaml, .json) cho các lượt huấn luyện, thuật toán.
│   └── utils/          #  Các hàm tiện ích, mã tái sử dụng (ví dụ: xử lý log, định dạng dữ liệu).
├── tools/              #  Các script/công cụ độc lập (ví dụ: tạo mạng lưới, phân tích dữ liệu thô).
├── results/            #  Nơi lưu trữ tất cả các kết quả đầu ra.
│   ├── models/         #  -> Các file checkpoints của mô hình đã huấn luyện (.pth, .zip).
│   ├── logs/           #  -> Các file log (TensorBoard, CSV) để theo dõi quá trình huấn luyện.
│   └── plots/          #  -> Các biểu đồ, hình ảnh được tạo ra từ quá trình đánh giá.
├── tests/              #  Chứa các bài kiểm thử đơn vị (unit tests) cho mã nguồn.
├── scripts/            #  Chứa các Script chính để CHẠY huấn luyện.
├── README.md           #  (Bạn đang đọc file này) Hướng dẫn tổng quan về dự án.
├── pyproject.toml      #  Danh sách các thư viện Python cần thiết.
└── .gitignore          #  Các file/thư mục mà Git sẽ bỏ qua (ví dụ: venv/, results/, __pycache__/).

## Đóng góp (Contributing)

Đây là dự án mã nguồn mở

## Giấy phép (License)
[Dự án này được cấp phép theo Giấy phép MIT]