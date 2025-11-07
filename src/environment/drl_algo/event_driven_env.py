"""Event-driven Environment wrapper.

This module wraps SumoEnvironment to work with EventBus,
enabling decoupled communication between Env and Simulator.
"""

import ray
import time
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import gymnasium as gym


@ray.remote
class EventDrivenEnvironment:
    """Ray actor wrapper for Environment that communicates via EventBus.
    
    This actor:
    1. Subscribes to "init", "step_result", "reset_result" topics
    2. Publishes to "action", "reset", "close" topics
    3. Provides standard Gym interface to RL agents
    4. Handles synchronization with simulator
    """
    
    def __init__(self, bus, env_id: str, simulator_id: str, timeout: float = 10.0):
        """Initialize event-driven environment.
        
        Args:
            bus: EventBus actor reference
            env_id: Unique ID for this environment
            simulator_id: ID of the simulator to communicate with
            timeout: Timeout for waiting for simulator responses (seconds)
        """
        self.bus = bus
        self.env_id = env_id
        self.simulator_id = simulator_id
        self.timeout = timeout
        
        # State
        self.initialized = False
        self.agent_ids = []
        self.current_observations = {}
        self.current_rewards = {}
        self.current_dones = {}
        self.current_info = {}
        
        # Response tracking
        self.waiting_for_response = False
        self.response_received = False
        self.response_data = None
        
    def subscribe_to_bus(self, self_ref):
        """Subscribe to relevant topics on the event bus.
        
        Args:
            self_ref: Reference to this actor (self)
        """
        # Subscribe to messages from Simulator
        ray.get([
            self.bus.subscribe.remote("init", self_ref, self.env_id),
            self.bus.subscribe.remote("step_result", self_ref, self.env_id),
            self.bus.subscribe.remote("reset_result", self_ref, self.env_id),
            self.bus.subscribe.remote("close_result", self_ref, self.env_id),
        ])
        print(f"🔌 {self.env_id}: Subscribed to EventBus")
    
    def on_message(self, sender_id: str, topic: str, **kwargs):
        """Handle incoming messages from EventBus.
        
        Args:
            sender_id: ID of message sender (should be simulator_id)
            topic: Topic name
            **kwargs: Message payload
        """
        # Only accept messages from our designated simulator
        if sender_id != self.simulator_id:
            return
        
        if topic == "init":
            self._handle_init(**kwargs)
        elif topic == "step_result":
            self._handle_step_result(**kwargs)
        elif topic == "reset_result":
            self._handle_reset_result(**kwargs)
        elif topic == "close_result":
            self._handle_close_result(**kwargs)
        else:
            print(f"⚠️ {self.env_id}: Unknown topic '{topic}'")
    
    def _handle_init(self, **kwargs):
        """Handle init complete message from Simulator."""
        success = kwargs.get("success", False)
        
        if success:
            self.agent_ids = kwargs.get("agent_ids", [])
            self.current_observations = kwargs.get("observations", {})
            self.initialized = True
            print(f"✅ {self.env_id}: Initialization complete, {len(self.agent_ids)} agents")
        else:
            error = kwargs.get("error", "Unknown error")
            print(f"❌ {self.env_id}: Initialization failed: {error}")
        
        # Mark response received
        self.response_received = True
        self.response_data = kwargs
    
    def _handle_step_result(self, **kwargs):
        """Handle step result message from Simulator."""
        success = kwargs.get("success", False)
        
        if success:
            self.current_observations = kwargs.get("observations", {})
            self.current_rewards = kwargs.get("rewards", {})
            self.current_dones = kwargs.get("dones", {})
            self.current_info = kwargs.get("info", {})
            print(f"📥 {self.env_id}: Received step_result")
        else:
            error = kwargs.get("error", "Unknown error")
            print(f"❌ {self.env_id}: Step failed: {error}")
        
        # Mark response received
        self.response_received = True
        self.response_data = kwargs
    
    def _handle_reset_result(self, **kwargs):
        """Handle reset result message from Simulator."""
        success = kwargs.get("success", False)
        
        if success:
            self.agent_ids = kwargs.get("agent_ids", [])
            self.current_observations = kwargs.get("observations", {})
            self.current_rewards = {aid: 0.0 for aid in self.agent_ids}
            self.current_dones = {aid: False for aid in self.agent_ids}
            self.current_dones["__all__"] = False
            self.current_info = {}
            print(f"🔄 {self.env_id}: Reset complete")
        else:
            error = kwargs.get("error", "Unknown error")
            print(f"❌ {self.env_id}: Reset failed: {error}")
        
        # Mark response received
        self.response_received = True
        self.response_data = kwargs
    
    def _handle_close_result(self, **kwargs):
        """Handle close result message from Simulator."""
        success = kwargs.get("success", False)
        
        if success:
            self.initialized = False
            print(f"🛑 {self.env_id}: Close complete")
        else:
            print(f"⚠️ {self.env_id}: Close had issues")
        
        # Mark response received
        self.response_received = True
        self.response_data = kwargs
    
    def _wait_for_response(self, timeout: Optional[float] = None) -> bool:
        """Wait for response from simulator.
        
        Args:
            timeout: Timeout in seconds (None = use default)
            
        Returns:
            True if response received, False if timeout
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while not self.response_received:
            if time.time() - start_time > timeout:
                print(f"⏱️ {self.env_id}: Timeout waiting for response")
                return False
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        return True
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment (sends reset message to simulator).
        
        Args:
            seed: Random seed
            
        Returns:
            Initial observations dict
        """
        # Clear response tracking
        self.response_received = False
        self.response_data = None
        
        # Send reset message
        self.bus.publish.remote(
            self.env_id,
            "reset",
            seed=seed
        )
        
        print(f"📤 {self.env_id}: Sent reset request")
        
        # Wait for response
        if not self._wait_for_response():
            raise TimeoutError(f"{self.env_id}: Timeout waiting for reset response")
        
        # Check if reset was successful
        if not self.response_data.get("success", False):
            error = self.response_data.get("error", "Unknown error")
            raise RuntimeError(f"{self.env_id}: Reset failed: {error}")
        
        return self.current_observations
    
    def step(self, actions: Union[Dict[str, int], int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step environment (sends action message to simulator).
        
        Args:
            actions: Actions dict or single action
            
        Returns:
            (observations, rewards, dones, info) tuple
        """
        if not self.initialized:
            raise RuntimeError(f"{self.env_id}: Environment not initialized, call reset() first")
        
        # Clear response tracking
        self.response_received = False
        self.response_data = None
        
        # Send action message
        self.bus.publish.remote(
            self.env_id,
            "action",
            actions=actions
        )
        
        print(f"📤 {self.env_id}: Sent action")
        
        # Wait for response
        if not self._wait_for_response():
            raise TimeoutError(f"{self.env_id}: Timeout waiting for step response")
        
        # Check if step was successful
        if not self.response_data.get("success", False):
            error = self.response_data.get("error", "Unknown error")
            raise RuntimeError(f"{self.env_id}: Step failed: {error}")
        
        return (
            self.current_observations,
            self.current_rewards,
            self.current_dones,
            self.current_info
        )
    
    def close(self):
        """Close environment (sends close message to simulator)."""
        # Clear response tracking
        self.response_received = False
        self.response_data = None
        
        # Send close message
        self.bus.publish.remote(
            self.env_id,
            "close"
        )
        
        print(f"📤 {self.env_id}: Sent close request")
        
        # Wait for response (shorter timeout for close)
        self._wait_for_response(timeout=2.0)
    
    def get_state(self):
        """Get current environment state (for debugging)."""
        return {
            "initialized": self.initialized,
            "agent_ids": self.agent_ids,
            "observations": self.current_observations,
            "rewards": self.current_rewards,
            "dones": self.current_dones,
            "info": self.current_info,
        }
