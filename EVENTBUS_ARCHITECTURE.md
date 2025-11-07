## Kiến trúc EventBus: Tách biệt Env và SUMO

### Tổng quan

Kiến trúc EventBus cho phép tách biệt hoàn toàn môi trường RL (Environment) và backend mô phỏng SUMO thông qua cơ chế publish-subscribe bất đồng bộ.

### Kiến trúc cũ (API trực tiếp)

```
Agent (RL) → Env → SimulatorAPI → SumoSimulator → SUMO
```

**Vấn đề:**
- Env và Simulator gắn chặt (tight coupling)
- Khó chạy phân tán
- Khó test riêng từng thành phần
- Không hỗ trợ nhiều Env với một Simulator

### Kiến trúc mới (EventBus)

```
Agent (RL) → EventDrivenEnv ──┐
                               ├──→ EventBus ──┐
Agent (RL) → EventDrivenEnv ──┘                ├──→ EventDrivenSimulator → SumoSimulator → SUMO
                                                │
Agent (RL) → EventDrivenEnv ────────────────────┘
```

**Ưu điểm:**
- ✅ Tách biệt hoàn toàn (loose coupling)
- ✅ Hỗ trợ nhiều Env với một Simulator
- ✅ Dễ chạy phân tán (Ray actors)
- ✅ Dễ test (mock EventBus)
- ✅ Log tập trung tất cả message
- ✅ Có thể thêm middleware (logging, monitoring, ...)

---

### Các thành phần

#### 1. **EventBus** (`src/sim/event_bus.py`)
- **Chức năng:** Message broker giữa Env và Simulator
- **Topics:**
  - `action`: Env → Simulator (chứa actions)
  - `step_result`: Simulator → Env (chứa obs/rewards/dones/info)
  - `reset`: Env → Simulator (yêu cầu reset)
  - `reset_result`: Simulator → Env (kết quả reset)
  - `close`: Env → Simulator (yêu cầu đóng)
  - `init`: Simulator → Env (khởi tạo hoàn tất)

#### 2. **EventDrivenSumoSimulator** (`src/sim/event_driven_simulator.py`)
- **Chức năng:** Wrapper cho SumoSimulator, giao tiếp qua EventBus
- **Subscribed topics:** `action`, `reset`, `close`
- **Published topics:** `init`, `step_result`, `reset_result`, `close_result`
- **Ray Actor:** Chạy trên Ray cluster

#### 3. **EventDrivenEnvironment** (`src/environment/drl_algo/event_driven_env.py`)
- **Chức năng:** Wrapper cho Environment, giao tiếp qua EventBus
- **Subscribed topics:** `init`, `step_result`, `reset_result`, `close_result`
- **Published topics:** `action`, `reset`, `close`
- **Interface:** Giống Gym.Env (reset, step, close)
- **Ray Actor:** Chạy trên Ray cluster

---

### Luồng giao tiếp

#### **Khởi tạo:**
```
1. Tạo EventBus
2. Tạo EventDrivenSumoSimulator, subscribe vào bus
3. Gọi simulator.initialize() → publish "init"
4. Tạo EventDrivenEnvironment, subscribe vào bus
5. Env nhận "init" → ready
```

#### **Reset:**
```
1. Agent gọi env.reset()
2. Env publish "reset" → EventBus
3. EventBus forward "reset" → Simulator
4. Simulator reset SUMO, publish "reset_result"
5. EventBus forward "reset_result" → Env
6. Env trả observations cho Agent
```

#### **Step:**
```
1. Agent gọi env.step(actions)
2. Env publish "action" → EventBus
3. EventBus forward "action" → Simulator
4. Simulator step SUMO, publish "step_result"
5. EventBus forward "step_result" → Env
6. Env trả (obs, rewards, dones, info) cho Agent
```

---

### Cách sử dụng

#### **Ví dụ cơ bản:**

```python
import ray
from src.sim.event_bus import EventBus
from src.sim.event_driven_simulator import EventDrivenSumoSimulator
from src.environment.drl_algo.event_driven_env import EventDrivenEnvironment

# Initialize Ray
ray.init()

# Create EventBus
bus = EventBus.remote()

# Create Simulator
simulator = EventDrivenSumoSimulator.remote(
    bus=bus,
    simulator_id="sim_1",
    net_file="network/grid4x4/grid4x4.net.xml",
    route_file="network/grid4x4/grid4x4.rou.xml",
    use_gui=False,
)
ray.get(simulator.subscribe_to_bus.remote(simulator))
ray.get(simulator.initialize.remote())

# Create Environment
env = EventDrivenEnvironment.remote(
    bus=bus,
    env_id="env_1",
    simulator_id="sim_1"
)
ray.get(env.subscribe_to_bus.remote(env))

# Use like normal Gym environment
observations = ray.get(env.reset.remote())
for _ in range(100):
    actions = {agent: 0 for agent in observations.keys()}
    obs, rewards, dones, info = ray.get(env.step.remote(actions))
    if dones["__all__"]:
        break

ray.get(env.close.remote())
ray.shutdown()
```

#### **Chạy demo:**

```bash
python examples/eventbus_demo.py
```

---

### So sánh với kiến trúc cũ

| Tiêu chí | API trực tiếp | EventBus |
|----------|---------------|----------|
| **Coupling** | Tight (gắn chặt) | Loose (tách biệt) |
| **Phân tán** | Khó | Dễ (Ray actors) |
| **Multi-Env** | Mỗi Env = 1 Simulator | Nhiều Env = 1 Simulator |
| **Testing** | Khó mock | Dễ mock EventBus |
| **Logging** | Phân tán | Tập trung tại bus |
| **Độ phức tạp** | Thấp | Trung bình |
| **Latency** | Thấp | Cao hơn (do message passing) |

---

### Khi nào nên dùng EventBus?

**Nên dùng khi:**
- ✅ Cần chạy phân tán (nhiều máy, nhiều GPU)
- ✅ Cần nhiều Env với một Simulator
- ✅ Cần log/monitor tập trung
- ✅ Cần test riêng từng thành phần
- ✅ Dự án lớn, nhiều người phát triển

**Không nên dùng khi:**
- ❌ Dự án nhỏ, đơn giản
- ❌ Cần latency thấp nhất (real-time)
- ❌ Chạy trên một máy đơn giản

---

### Mở rộng

#### **Thêm logging middleware:**
```python
@ray.remote
class LoggingMiddleware:
    def __init__(self, bus):
        self.bus = bus
        # Subscribe to all topics
        
    def on_message(self, sender, topic, **kwargs):
        # Log message
        print(f"[LOG] {sender} → {topic}: {kwargs}")
```

#### **Thêm monitoring:**
```python
# Monitor latency, throughput, error rate
# Publish metrics to EventBus topic "metrics"
```

#### **Multi-simulator:**
```python
# Create multiple simulators for different scenarios
sim1 = EventDrivenSumoSimulator.remote(bus, "sim_1", ...)
sim2 = EventDrivenSumoSimulator.remote(bus, "sim_2", ...)

# Envs can choose which simulator to use
env1 = EventDrivenEnvironment.remote(bus, "env_1", simulator_id="sim_1")
env2 = EventDrivenEnvironment.remote(bus, "env_2", simulator_id="sim_2")
```

---

### Tài liệu tham khảo

- File demo: `examples/eventbus_demo.py`
- EventBus implementation: `src/sim/event_bus.py`
- Event-driven simulator: `src/sim/event_driven_simulator.py`
- Event-driven environment: `src/environment/drl_algo/event_driven_env.py`
- Ray documentation: https://docs.ray.io/

---

**Lưu ý:** Kiến trúc EventBus phức tạp hơn API trực tiếp, nhưng mang lại nhiều lợi ích cho hệ thống lớn. Bạn có thể giữ cả hai kiến trúc và chọn sử dụng tùy theo nhu cầu.
