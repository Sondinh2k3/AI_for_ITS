"""Event Bus for communication between Environment and Simulator.

This module provides a publish-subscribe event bus for decoupling
the RL environment from the SUMO simulator backend.
"""

import ray
from collections import defaultdict
from typing import Any, Dict, List, Optional
import time

# Tạo ray actor, cho phép chạy các class phân tán, song song trên nhiều process/máy.
# Các ray actor được định nghĩa dưới dạng class
# Dưới đây là một Ray actor được định nghĩa: đó là class EventBus. Mỗi khi khởi tạo một instance của class này, sẽ tạo ra một actor có khả năng chạy độc lập, các actor sẽ chạy song song với nhau

@ray.remote
class EventBus:
    """Central event bus for managing communication between Env and Simulator.
    
    The EventBus acts as a message broker between:
    - Environment (Env): Sends actions, receives observations/rewards
    - Simulator (SUMO): Receives actions, sends observations/rewards
    
    Message Topics:
    - "action": Env → Simulator (contains agent actions)
    - "observation": Simulator → Env (contains observations for agents)
    - "reward": Simulator → Env (contains rewards for agents)
    - "reset": Env → Simulator (request reset)
    - "close": Env → Simulator (request shutdown)
    - "init": Simulator → Env (initialization complete)
    """
    
    def __init__(self):
        """Initialize event bus with empty subscriber lists."""
        # topic -> [(subscriber_id, actor_ref)]
        self.subscribers = defaultdict(list)
        # topic -> List[message dict]
        self.message_cache = defaultdict(list)
        
    def subscribe(self, topic: str, subscriber_ref, subscriber_id: str):
        """Subscribe to a topic.
        
        Args:
            topic: Topic name (e.g., "action", "observation")
            subscriber_ref: Ray actor reference => Tham chieu toi actor se nhan message
            subscriber_id: Unique ID for subscriber => Mỗi subcriber sẽ có một định danh duy nhất
        """
        self.subscribers[topic].append((subscriber_id, subscriber_ref))
        print(f" EventBus: {subscriber_id} subscribed to '{topic}'")
        return f"{subscriber_id} subscribed to {topic}"
    
    def publish(self, sender_id: str, topic: str, **kwargs):
        """Publish message to all subscribers of a topic.
        
        Args:
            sender_id: ID of the sender
            topic: Topic name
            **kwargs: Message payload
            
        Returns:
            Number of receivers notified
        """
        receivers = 0
        
        # Cache message for synchronous retrieval (append to queue)
        self.message_cache[topic].append(
            {
                "sender_id": sender_id,
                "payload": kwargs,
                "timestamp": time.time(),
            }
        )
        
        if topic not in self.subscribers:
            print(f" EventBus: No subscribers for topic '{topic}'")
            return 0
        
        # Send to all subscribers except sender
        for sub_id, ref in self.subscribers[topic]:
            if sub_id != sender_id:
                ref.on_message.remote(sender_id, topic, **kwargs)
                receivers += 1
        
        return receivers
    
    def get_latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get latest (most recent) message from topic without removing it."""
        queue = self.message_cache.get(topic)
        if not queue:
            return None
        return queue[-1]

    def get_message(self, topic: str, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve and remove the first message for a topic.
        
        If request_id is provided, returns the first message whose payload contains
        a matching request_id. Otherwise returns the oldest message in the queue.
        """
        queue = self.message_cache.get(topic)
        if not queue:
            return None

        if request_id is None:
            return queue.pop(0)

        for idx, message in enumerate(queue):
            if message["payload"].get("request_id") == request_id:
                return queue.pop(idx)

        return None
    
    def clear_cache(self, topic: Optional[str] = None):
        """Clear message cache.
        
        Args:
            topic: Topic to clear, or None to clear all
        """
        if topic is None:
            self.message_cache.clear()
        elif topic in self.message_cache:
            self.message_cache[topic].clear()
    
    def list_topics(self):
        """List all topics and their subscribers."""
        return {
            topic: [sub_id for sub_id, _ in subs]
            for topic, subs in self.subscribers.items()
        }
    
    def unsubscribe(self, topic: str, subscriber_id: str):
        """Unsubscribe from a topic.
        
        Args:
            topic: Topic name
            subscriber_id: ID of subscriber to remove
        """
        if topic in self.subscribers:
            self.subscribers[topic] = [
                (sid, ref) for sid, ref in self.subscribers[topic]
                if sid != subscriber_id
            ]
            print(f"❌ EventBus: {subscriber_id} unsubscribed from '{topic}'")
