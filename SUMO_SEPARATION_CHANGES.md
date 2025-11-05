# Tách Logic SUMO Simulation khỏi Environment

## Tóm tắt

Đã tách toàn bộ logic xử lý SUMO (khởi chạy, bước mô phỏng, tắt) từ `env.py` sang `SumoSim.py`. Điều này làm code sạch hơn, dễ test hơn, và tách biệt rõ ràng giữa:
- **SumoSim**: Quản lý SUMO simulator (traci connection, start/step/close)
- **SumoEnvironment**: Quản lý RL environment (observations, rewards, actions)

## Thay đổi chi tiết

### 1. **File: `src/sim/Sumo_sim.py`** (mở rộng)

#### A. Thêm tham số vào `__init__`
```python
def __init__(
    self,
    net_file: str,
    route_file: Optional[str],
    label: str = "0",
    use_gui: bool = False,
    virtual_display: Optional[tuple] = None,
    begin_time: int = 0,
    sumo_seed: Optional[str] = None,
    additional_sumo_cmd: Optional[List[str]] = None,
    no_warnings: bool = False,
    max_depart_delay: int = -1,        # ← Mới
    waiting_time_memory: int = 1000,   # ← Mới
    time_to_teleport: int = -1,        # ← Mới
):
```
**Mục đích:** SumoSim giờ có đủ thông tin để xây dựng command SUMO hoàn chỉnh.

#### B. Cập nhật `_build_cmd()`
Thêm xử lý cho các tham số traffic-related (chỉ khi không phải net_only):
```python
if not net_only:
    if self.max_depart_delay >= 0:
        cmd.extend(["--max-depart-delay", str(self.max_depart_delay)])
    if self.waiting_time_memory > 0:
        cmd.extend(["--waiting-time-memory", str(self.waiting_time_memory)])
    if self.time_to_teleport >= 0:
        cmd.extend(["--time-to-teleport", str(self.time_to_teleport)])
```

#### C. Thêm method `_setup_gui_display()`
Tách logic setup GUI schema:
```python
def _setup_gui_display(self):
    """Setup GUI display settings for SUMO-GUI."""
    try:
        if "DEFAULT_VIEW" not in dir(traci.gui):
            traci.gui.DEFAULT_VIEW = "View #0"
        self.conn.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
    except Exception:
        pass
```

#### D. Sửa `start()`
Gọi `_setup_gui_display()` sau khi start:
```python
def start(self):
    cmd = self._build_cmd(gui=self.use_gui, net_only=False)
    
    if self.use_gui and self.virtual_display and not LIBSUMO:
        self.disp = SmartDisplay(size=self.virtual_display)
        self.disp.start()
    
    if LIBSUMO:
        traci.start(cmd)
        self.conn = traci
    else:
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
    
    self._started = True
    
    if self.use_gui or self.virtual_display:
        self._setup_gui_display()
    
    return self.conn
```

---

### 2. **File: `src/environment/drl_algo/env.py`** (tách logic)

#### A. Xóa `_sumo_binary` từ `__init__`
**Trước:**
```python
if self.use_gui or self.render_mode is not None:
    self._sumo_binary = sumolib.checkBinary("sumo-gui")
else:
    self._sumo_binary = sumolib.checkBinary("sumo")
```
**Sau:** Xóa hoàn toàn, SumoSim tự xử lý.

#### B. Sửa `__init__` để tạo và sử dụng `SumoSim`
**Trước:**
```python
if LIBSUMO:
    traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
    conn = traci
else:
    traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
    conn = traci.getConnection("init_connection" + self.label)
```
**Sau:**
```python
self.sumo_sim = SumoSim(
    net_file=self._net,
    route_file=None,
    label=self.label,
    use_gui=self.use_gui,
    virtual_display=self.virtual_display,
    begin_time=self.begin_time,
    sumo_seed=str(self.sumo_seed) if self.sumo_seed != "random" else "random",
    additional_sumo_cmd=self.additional_sumo_cmd.split() if self.additional_sumo_cmd else [],
    no_warnings=not self.sumo_warnings,
    max_depart_delay=self.max_depart_delay,
    waiting_time_memory=self.waiting_time_memory,
    time_to_teleport=self.time_to_teleport,
)

conn = self.sumo_sim.start_temp()
```

#### C. Hoàn toàn sửa lại `_start_simulation()`
**Trước:** 200+ dòng xây dựng command SUMO thủ công
**Sau:** 5 dòng gọi SumoSim
```python
def _start_simulation(self):
    """Start the SUMO simulation via SumoSim runner."""
    self.sumo_sim.route_file = self._route
    self.sumo = self.sumo_sim.start()
```

#### D. Sửa `_sumo_step()` để gọi `SumoSim.step()`
**Trước:**
```python
self.sumo.simulationStep()
```
**Sau:**
```python
self.sumo_sim.step()
```

#### E. Sửa `close()` để gọi `SumoSim.close()`
**Trước:** 10 dòng xử lý traci.close/switch/display.stop
**Sau:** 2 dòng
```python
def close(self):
    if self.sumo_sim is not None:
        self.sumo_sim.close()
    self.sumo = None
```

---

## Lợi ích

✅ **Tách trách nhiệm rõ ràng:**
  - SumoSim: Simulator management (start, step, close, connection)
  - SumoEnvironment: RL environment (observation, reward, action)

✅ **Dễ test:**
  - Mock SumoSim để test env logic mà không cần SUMO
  - Test SumoSim riêng biệt

✅ **Dễ tái sử dụng:**
  - SumoSim có thể dùng độc lập với các project khác
  - Không phụ thuộc vào RL framework

✅ **Code sạch hơn:**
  - env.py từ 600+ dòng → dễ theo dõi logic RL
  - Logic SUMO tập trung, dễ bảo trì

✅ **Mở rộng dễ:**
  - Thêm render method vào SumoSim
  - Thêm multi-simulation support

---

## Kiểm tra

Mọi lỗi import/syntax đã được kiểm tra — không có lỗi.

**Cần test thêm:**
- Reset environment → start SUMO
- Step environment → call sumo_sim.step()
- Close environment → proper cleanup
- Multi-agent scenarios

---

## File liên quan

- `src/sim/Sumo_sim.py` — Simulator runner (mở rộng)
- `src/environment/drl_algo/env.py` — RL environment (tách logic)
- `src/environment/drl_algo/traffic_signal.py` — Không thay đổi
