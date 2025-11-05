# Kế Hoạch Tách Biệt Mô Phỏng SUMO khỏi Environment

## Tóm Tắt

Chúng ta đang thực hiện quá trình tách hoàn toàn mô phỏng SUMO ra khỏi `env.py` thông qua một **API giao tiếp rõ ràng**. Sau khi hoàn tất:

- **`env.py`**: Chỉ chứa logic RL, không biết gì về SUMO, không import `traci` hay `sumolib`
- **`Sumo_sim.py`**: Chứa toàn bộ SUMO-specific logic, implement `SimulatorAPI`
- **`simulator_api.py`**: Định nghĩa abstract API interface

---

## Kiến Trúc Mới

```
┌────────────────────┐
│  env.py            │  (RL Environment - Gym/PettingZoo)
│  - step()          │  - Nhận action từ agents
│  - reset()         │  - Gọi simulator.step(actions)
│  - close()         │  - Xử lý observations, rewards
│  ✅ Không import    │  - Không biết SUMO
│     traci/sumolib  │
└─────────┬──────────┘
          │
          │ Giao tiếp qua SimulatorAPI
          │
┌─────────▼──────────────────┐
│  Sumo_sim.py               │  (SUMO Simulator Backend)
│  class SumoSimulator        │  - Implement SimulatorAPI
│  (implement SimulatorAPI)   │  - Quản lý SUMO connection
│  - initialize()            │  - Quản lý TrafficSignals
│  - step()                  │  - Tính observation/reward
│  - reset()                 │  - Tất cả traci API calls
│  - close()                 │
│  - get_agent_ids()         │
│  ✅ Duy nhất import traci   │
└────────────────────────────┘
          ▲
          │
┌─────────┴──────────────────┐
│  simulator_api.py          │  (Abstract Interface)
│  class SimulatorAPI        │  - Define required methods
│  (abstract base class)      │
└────────────────────────────┘
```

---

## Các Bước Thực Hiện

### **Bước 1: ✅ Tạo API Abstract (Hoàn Tất)**
- File: `src/sim/simulator_api.py`
- Định nghĩa class `SimulatorAPI` với các abstract methods:
  - `initialize()` — khởi tạo mô phỏng
  - `step(actions)` — tiến mô phỏng 1 bước
  - `reset()` — reset về trạng thái ban đầu
  - `close()` — đóng mô phỏng
  - `get_agent_ids()` — lấy danh sách agents
  - `get_observation_space(agent_id)` — lấy obs space
  - `get_action_space(agent_id)` — lấy action space
  - `get_sim_step()` — lấy thời gian mô phỏng hiện tại

### **Bước 2: ✅ Tạo Khung SumoSimulator (Hoàn Tất)**
- File: `src/sim/Sumo_sim.py`
- Class `SumoSimulator(SimulatorAPI)` implement toàn bộ interface
- Hiện tại là khung (skeleton) với `NotImplementedError`
- **TODO**: Implement toàn bộ các methods

### **Bước 3: 🔄 Sửa env.py để sử dụng SumoSimulator (Chưa Làm)**
- Import `SumoSimulator` thay vì `traci`, `sumolib`
- Thay `self.sumo_sim` bằng `self.simulator`
- Tất cả giao tiếp với SUMO đều qua `self.simulator.method_name()`
- Xóa tất cả import SUMO-related từ env.py

### **Bước 4: 🔄 Implement toàn bộ methods trong SumoSimulator (Chưa Làm)**
- Di chuyển logic từ env.py sang SumoSimulator
- Implement:
  - `_start_temp_connection()` — khởi tạo tạm để đọc metadata
  - `_start_full_connection()` — khởi tạo đầy đủ
  - `_build_traffic_signals()` — tạo các TrafficSignal objects
  - `_close_connection()` — đóng kết nối
  - Implement toàn bộ abstract methods từ SimulatorAPI

---

## Chi Tiết Từng Bước Cần Làm

### **Bước 3: Sửa env.py**

**Trước:**
```python
import traci
import sumolib

class SumoEnvironment(gym.Env):
    def __init__(self, ...):
        self.sumo_sim = SumoSim(...)
        conn = self.sumo_sim.start_temp()
        self.ts_ids = list(conn.trafficlight.getIDList())
        # ...
```

**Sau:**
```python
from ...sim.Sumo_sim import SumoSimulator

class SumoEnvironment(gym.Env):
    def __init__(self, ...):
        self.simulator = SumoSimulator(
            net_file=self._net,
            route_file=self._route,
            # ...tất cả parameters...
        )
        initial_obs = self.simulator.initialize()
        self.ts_ids = self.simulator.get_agent_ids()
        # ...
```

**Thay đổi toàn bộ env.py:**
- Xóa imports: `traci`, `sumolib`, `LIBSUMO`
- Xóa hàm: `_start_simulation()`, `_sumo_step()`, `_get_system_info()`, ...
- Sửa `step()`:
  ```python
  def step(self, actions):
      obs, rewards, dones, info = self.simulator.step(actions)
      # Xử lý RL logic nếu cần
      return obs, rewards, dones, info
  ```
- Sửa `reset()`:
  ```python
  def reset(self):
      obs = self.simulator.reset()
      # Khởi tạo RL state
      return obs
  ```
- Sửa `close()`:
  ```python
  def close(self):
      self.simulator.close()
  ```

### **Bước 4: Implement SumoSimulator**

Chuyển toàn bộ logic liên quan đến SUMO từ env.py sang SumoSimulator:

**Các hàm cần implement:**
1. `initialize()` — khởi tạo mô phỏng, lấy agent IDs, build traffic signals
2. `step(actions)` — áp dụng action, run SUMO, lấy obs/reward
3. `reset()` — đóng simulation cũ, start lại
4. `close()` — cleanup
5. `_start_temp_connection()` — start tạm connection
6. `_start_full_connection()` — start full connection
7. `_close_connection()` — close connection
8. `_build_traffic_signals()` — tạo TrafficSignal objects

---

## Lợi Ích

✅ **Tách trách nhiệm rõ ràng**
- `env.py`: RL environment logic only
- `Sumo_sim.py`: Simulator backend logic only

✅ **Dễ kiểm thử**
- Mock `SumoSimulator` để test env mà không cần SUMO
- Test SumoSimulator riêng biệt

✅ **Dễ thay thế backend**
- Có thể tạo `CityFlowSimulator(SimulatorAPI)` mà không thay đổi env.py

✅ **Code sạch hơn**
- env.py không còn 600+ dòng SUMO-specific code
- Dễ đọc, dễ bảo trì

---

## Checklist

- [x] Tạo `simulator_api.py` với abstract class
- [x] Tạo khung `SumoSimulator(SimulatorAPI)` trong `Sumo_sim.py`
- [ ] **TODO**: Implement toàn bộ methods trong SumoSimulator
- [ ] **TODO**: Sửa env.py để chỉ sử dụng SimulatorAPI
- [ ] **TODO**: Kiểm thử lại toàn bộ

---

## Tài Liệu Tham Khảo

- `src/sim/simulator_api.py` — Abstract API
- `src/sim/Sumo_sim.py` — SUMO implementation
- `src/environment/drl_algo/env.py` — RL environment (sau khi sửa)

---

## Ghi Chú

- Hiện tại `SumoSimulator` là khung với `NotImplementedError`
- Các import từ `traffic_signal` và `observations` có lỗi (sẽ fix sau)
- Quá trình này sẽ hoàn tất trong các bước tiếp theo
