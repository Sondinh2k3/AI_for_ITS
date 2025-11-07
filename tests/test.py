# ray_observer_group_dispatch.py
import ray
import time
from collections import defaultdict


# ---------------- EventBus ----------------
@ray.remote
class EventBus:
    def __init__(self):
        self.groups = defaultdict(list)

    def subscribe(self, group_name: str, worker_ref, worker_id: str):
        self.groups[group_name].append((worker_id, worker_ref))
        print(f"✅ {worker_id} subscribed to '{group_name}'")
        return f"{worker_id} subscribed to {group_name}"

    def publish(self, sender_id: str, group_name: str, **kwargs):
        """Broadcast message to all in the group (except sender)."""
        receivers = 0
        if group_name not in self.groups:
            print(f"⚠️ Group '{group_name}' không tồn tại.")
            return 0
        for wid, ref in self.groups[group_name]:
            if wid != sender_id:
                ref.on_message.remote(sender_id, group_name, **kwargs)
                receivers += 1
        return receivers

    def list_groups(self):
        return {g: [wid for wid, _ in members] for g, members in self.groups.items()}


# ---------------- Worker ----------------
@ray.remote
class Worker:
    def __init__(self, worker_id: str, bus):
        self.worker_id = worker_id
        self.bus = bus
        self.inbox = []

    def join_group(self, group_name: str, self_ref):
        return ray.get(self.bus.subscribe.remote(group_name, self_ref, self.worker_id))

    # --- Dynamic message routing ---
    def on_message(self, sender_id: str, group_name: str, **kwargs):
        ts = time.strftime("%H:%M:%S")
        self.inbox.append((group_name, sender_id, kwargs, ts))

        # Dynamically find handler
        handler_name = f"handle_{group_name}"
        handler = getattr(self, handler_name, None)
        if callable(handler):
            handler(sender_id, **kwargs)
        else:
            self.handle_default(sender_id, group_name, **kwargs)

    # --- Example Handlers (custom APIs) ---
    def handle_alpha(self, sender_id, **kwargs):
        """Xử lý message từ group 'alpha'"""
        print(f"🧩 [{self.worker_id}] API α: nhận từ {sender_id} -> {kwargs}")

    def handle_beta(self, sender_id, text=None):
        """Xử lý message từ group 'beta'"""
        print(f"🔧 [{self.worker_id}] API β: từ {sender_id} - nội dung='{text}'")

    def handle_gamma(self, sender_id, payload=[], test="None"):
        """Xử lý message từ group 'gamma'"""
        print(f"📊 [{self.worker_id}] API γ: dữ liệu: {sender_id} = {payload}, test={test}")

    def handle_default(self, sender_id, group_name, **kwargs):
        """Fallback khi không có handler phù hợp."""
        print(f"⚙️ [{self.worker_id}] Không có API cho group '{group_name}' — message={kwargs}")

    # --- Gửi message ---
    def send_to_group(self, group_name: str, **kwargs):
        receivers = ray.get(self.bus.publish.remote(self.worker_id, group_name, **kwargs))
        print(f"📤 {self.worker_id} -> '{group_name}' ({receivers} receivers): {kwargs}")
        return receivers

    def get_inbox(self):
        return list(self.inbox)


# ---------------- Demo ----------------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    bus = EventBus.remote()

    # 5 worker
    A = Worker.remote("A", bus)
    B = Worker.remote("B", bus)
    C = Worker.remote("C", bus)
    D = Worker.remote("D", bus)
    E = Worker.remote("E", bus)

    # Join groups
    ray.get([
        A.join_group.remote("alpha", A),
        B.join_group.remote("alpha", B),
        C.join_group.remote("beta", C),
        D.join_group.remote("alpha", D),
        D.join_group.remote("beta", D),
        E.join_group.remote("gamma", E),
    ])

    print("\n📋 Group membership:")
    print(ray.get(bus.list_groups.remote()))

    # Send messages with group-specific payloads
    A.send_to_group.remote("alpha", text="Sync request", step=1)
    C.send_to_group.remote("beta", text="Loss=0.01")
    D.send_to_group.remote("alpha", cmd="update", version="v2")
    E.send_to_group.remote("gamma", payload=[10, 20, 30])
    A.send_to_group.remote("gamma", test="cross-group test")  # no handler for A on gamma

    time.sleep(1)
    ray.shutdown()
