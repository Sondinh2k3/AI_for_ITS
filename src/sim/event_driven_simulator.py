"""Event-driven SUMO Simulator wrapper.

This module wraps SumoSimulator to work with EventBus,
enabling decoupled communication between Env and Simulator.
"""

import ray
from typing import Dict, Any, Optional
from .Sumo_sim import SumoSimulator


@ray.remote
class EventDrivenSumoSimulator:
    """Ray actor wrapper for SumoSimulator that communicates via EventBus.
    
    This actor:
    1. Subscribes to "action" and "reset" topics
    2. Publishes to "observation", "reward", and "init" topics
    3. Wraps a SumoSimulator instance
    4. Handles all SUMO-specific operations
    """
    
    def __init__(self, bus, simulator_id: str, **sumo_config):
        """Initialize event-driven simulator.
        
        Args:
            bus: EventBus actor reference
            simulator_id: Unique ID for this simulator
            **sumo_config: Configuration for SumoSimulator
        """
        self.bus = bus
        self.simulator_id = simulator_id
        self.sumo_config = sumo_config
        
        # Create underlying SUMO simulator
        self.simulator = SumoSimulator(**sumo_config)
        
        # State
        self.initialized = False
        self.current_observations = {}
        self.current_rewards = {}
        self.current_dones = {}
        self.current_info = {}
        self.current_metrics = {}
        self.current_system_info = {}
        self.current_per_agent_info = {}
        self.current_sim_step = 0.0
        
    def subscribe_to_bus(self, self_ref):
        """Subscribe to relevant topics on the event bus.
        
        Args:
            self_ref: Reference to this actor (self)
        """
        # Subscribe to action and reset messages from Env
        ray.get([
            self.bus.subscribe.remote("action", self_ref, self.simulator_id),
            self.bus.subscribe.remote("reset", self_ref, self.simulator_id),
            self.bus.subscribe.remote("close", self_ref, self.simulator_id),
            self.bus.subscribe.remote("init_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("observation_space_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("action_space_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("metrics_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("system_info_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("per_agent_info_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("sim_step_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("rgb_array_request", self_ref, self.simulator_id),
            self.bus.subscribe.remote("attribute_update", self_ref, self.simulator_id),
        ])
        print(f"🔌 {self.simulator_id}: Subscribed to EventBus")
    
    def initialize(self):
        """Initialize SUMO simulator and publish init complete."""
        try:
            # Initialize underlying simulator
            initial_obs = self.simulator.initialize()
            self.initialized = True
            
            # Publish initialization complete with initial observations
            self.bus.publish.remote(
                self.simulator_id,
                "init",
                observations=initial_obs,
                agent_ids=self.simulator.get_agent_ids(),
                success=True
            )
            
            print(f"✅ {self.simulator_id}: Initialized and published 'init'")
            return {"success": True, "observations": initial_obs}
            
        except Exception as e:
            print(f"❌ {self.simulator_id}: Initialization failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "init",
                success=False,
                error=str(e)
            )
            return {"success": False, "error": str(e)}
    
    def on_message(self, sender_id: str, topic: str, **kwargs):
        """Handle incoming messages from EventBus.
        
        Args:
            sender_id: ID of message sender
            topic: Topic name
            **kwargs: Message payload
        """
        target_simulator = kwargs.get("target_simulator")
        if target_simulator and target_simulator != self.simulator_id:
            return

        if topic == "action":
            self._handle_action(sender_id, **kwargs)
        elif topic == "reset":
            self._handle_reset(sender_id, **kwargs)
        elif topic == "close":
            self._handle_close(sender_id, **kwargs)
        elif topic == "init_request":
            self._handle_init_request(sender_id, **kwargs)
        elif topic == "observation_space_request":
            self._handle_observation_space_request(sender_id, **kwargs)
        elif topic == "action_space_request":
            self._handle_action_space_request(sender_id, **kwargs)
        elif topic == "metrics_request":
            self._handle_metrics_request(sender_id, **kwargs)
        elif topic == "system_info_request":
            self._handle_system_info_request(sender_id, **kwargs)
        elif topic == "per_agent_info_request":
            self._handle_per_agent_info_request(sender_id, **kwargs)
        elif topic == "sim_step_request":
            self._handle_sim_step_request(sender_id, **kwargs)
        elif topic == "rgb_array_request":
            self._handle_rgb_array_request(sender_id, **kwargs)
        elif topic == "attribute_update":
            self._handle_attribute_update(sender_id, **kwargs)
        else:
            print(f"⚠️ {self.simulator_id}: Unknown topic '{topic}'")
    
    def _handle_action(self, sender_id: str, **kwargs):
        """Handle action message from Env.
        
        Args:
            sender_id: ID of sender (Env)
            **kwargs: Should contain 'actions' dict
        """
        if not self.initialized:
            print(f"⚠️ {self.simulator_id}: Received action before initialization")
            return

        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        actions = kwargs.get("actions", {})
        
        try:
            # Step simulator
            observations, rewards, dones, info = self.simulator.step(actions)
            
            # Store current state
            self.current_observations = observations
            self.current_rewards = rewards
            self.current_dones = dones
            self.current_info = info
            self.current_metrics = getattr(self.simulator, "get_metrics", lambda: {} )()
            self.current_system_info = getattr(self.simulator, "get_system_info", lambda: {} )()
            self.current_per_agent_info = getattr(self.simulator, "get_per_agent_info", lambda: {} )()
            self.current_sim_step = info.get("step", 0.0)
            
            # Publish results back to Env
            self.bus.publish.remote(
                self.simulator_id,
                "step_result",
                env_id=env_id,
                request_id=request_id,
                observations=observations,
                rewards=rewards,
                dones=dones,
                info=info,
                metrics=self.current_metrics,
                system_info=self.current_system_info,
                per_agent_info=self.current_per_agent_info,
                success=True
            )
            
            print(f"📤 {self.simulator_id}: Published step_result")
            
        except Exception as e:
            print(f"❌ {self.simulator_id}: Step failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "step_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e)
            )
    
    def _handle_reset(self, sender_id: str, **kwargs):
        """Handle reset message from Env.
        
        Args:
            sender_id: ID of sender (Env)
            **kwargs: Optional reset parameters
        """
        try:
            env_id = kwargs.get("env_id") or sender_id
            request_id = kwargs.get("request_id")
            seed = kwargs.get("seed")
            if seed is not None:
                try:
                    self.simulator.sumo_seed = seed
                except AttributeError:
                    pass
                except Exception:
                    pass

            # Reset simulator
            initial_obs = self.simulator.reset()
            self.initialized = True
            
            # Clear cached state
            self.current_observations = {}
            self.current_rewards = {}
            self.current_dones = {}
            self.current_info = {}
            self.current_metrics = {}
            self.current_system_info = {}
            self.current_per_agent_info = {}
            self.current_sim_step = 0.0
            
            # Publish reset complete
            self.bus.publish.remote(
                self.simulator_id,
                "reset_result",
                env_id=env_id,
                request_id=request_id,
                observations=initial_obs,
                agent_ids=self.simulator.get_agent_ids(),
                success=True
            )
            
            print(f"🔄 {self.simulator_id}: Reset complete, published 'reset_result'")
            
        except Exception as e:
            print(f"❌ {self.simulator_id}: Reset failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "reset_result",
                env_id=kwargs.get("env_id") or sender_id,
                request_id=kwargs.get("request_id"),
                success=False,
                error=str(e)
            )
    
    def _handle_close(self, sender_id: str, **kwargs):
        """Handle close message from Env.
        
        Args:
            sender_id: ID of sender (Env)
            **kwargs: Optional close parameters
        """
        try:
            env_id = kwargs.get("env_id") or sender_id
            request_id = kwargs.get("request_id")
            self.simulator.close()
            self.initialized = False
            
            self.bus.publish.remote(
                self.simulator_id,
                "close_result",
                env_id=env_id,
                request_id=request_id,
                success=True
            )
            
            print(f"🛑 {self.simulator_id}: Closed successfully")
            
        except Exception as e:
            print(f"❌ {self.simulator_id}: Close failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "close_result",
                env_id=kwargs.get("env_id") or sender_id,
                request_id=kwargs.get("request_id"),
                success=False,
                error=str(e)
            )
    
    def _handle_init_request(self, sender_id: str, **kwargs):
        """Handle initialization request from Env."""
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            if not self.initialized:
                initial_obs = self.simulator.initialize()
                self.initialized = True
            else:
                initial_obs = self.current_observations or self.simulator.initialize()
            agent_ids = self.simulator.get_agent_ids()

            self.bus.publish.remote(
                self.simulator_id,
                "init_result",
                env_id=env_id,
                request_id=request_id,
                observations=initial_obs,
                agent_ids=agent_ids,
                success=True,
            )
        except Exception as e:
            print(f"❌ {self.simulator_id}: Init request failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "init_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_observation_space_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        agent_id = kwargs.get("agent_id")
        try:
            space = self.simulator.get_observation_space(agent_id)
            self.bus.publish.remote(
                self.simulator_id,
                "observation_space_result",
                env_id=env_id,
                request_id=request_id,
                agent_id=agent_id,
                space=space,
                success=True,
            )
        except Exception as e:
            print(f"❌ {self.simulator_id}: Observation space request failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "observation_space_result",
                env_id=env_id,
                request_id=request_id,
                agent_id=agent_id,
                success=False,
                error=str(e),
            )

    def _handle_action_space_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        agent_id = kwargs.get("agent_id")
        try:
            space = self.simulator.get_action_space(agent_id)
            self.bus.publish.remote(
                self.simulator_id,
                "action_space_result",
                env_id=env_id,
                request_id=request_id,
                agent_id=agent_id,
                space=space,
                success=True,
            )
        except Exception as e:
            print(f"❌ {self.simulator_id}: Action space request failed: {e}")
            self.bus.publish.remote(
                self.simulator_id,
                "action_space_result",
                env_id=env_id,
                request_id=request_id,
                agent_id=agent_id,
                success=False,
                error=str(e),
            )

    def _handle_metrics_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            metrics = getattr(self.simulator, "get_metrics", lambda: {} )()
            self.bus.publish.remote(
                self.simulator_id,
                "metrics_result",
                env_id=env_id,
                request_id=request_id,
                metrics=metrics,
                success=True,
            )
        except Exception as e:
            self.bus.publish.remote(
                self.simulator_id,
                "metrics_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_system_info_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            system_info = getattr(self.simulator, "get_system_info", lambda: {} )()
            self.bus.publish.remote(
                self.simulator_id,
                "system_info_result",
                env_id=env_id,
                request_id=request_id,
                system_info=system_info,
                success=True,
            )
        except Exception as e:
            self.bus.publish.remote(
                self.simulator_id,
                "system_info_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_per_agent_info_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            per_agent_info = getattr(self.simulator, "get_per_agent_info", lambda: {} )()
            self.bus.publish.remote(
                self.simulator_id,
                "per_agent_info_result",
                env_id=env_id,
                request_id=request_id,
                per_agent_info=per_agent_info,
                success=True,
            )
        except Exception as e:
            self.bus.publish.remote(
                self.simulator_id,
                "per_agent_info_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_sim_step_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            step = getattr(self.simulator, "get_sim_step", lambda: 0.0)()
            self.bus.publish.remote(
                self.simulator_id,
                "sim_step_result",
                env_id=env_id,
                request_id=request_id,
                step=step,
                success=True,
            )
        except Exception as e:
            self.bus.publish.remote(
                self.simulator_id,
                "sim_step_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_rgb_array_request(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        try:
            frame = getattr(self.simulator, "get_rgb_array", lambda: None)()
            self.bus.publish.remote(
                self.simulator_id,
                "rgb_array_result",
                env_id=env_id,
                request_id=request_id,
                frame=frame,
                success=True,
            )
        except Exception as e:
            self.bus.publish.remote(
                self.simulator_id,
                "rgb_array_result",
                env_id=env_id,
                request_id=request_id,
                success=False,
                error=str(e),
            )

    def _handle_attribute_update(self, sender_id: str, **kwargs):
        env_id = kwargs.get("env_id") or sender_id
        request_id = kwargs.get("request_id")
        attribute = kwargs.get("attribute")
        value = kwargs.get("value")
        success = True
        error = None
        try:
            if attribute and hasattr(self.simulator, attribute):
                setattr(self.simulator, attribute, value)
            else:
                success = False
                error = f"Unknown attribute '{attribute}'"
        except Exception as exc:
            success = False
            error = str(exc)

        response_payload = {
            "env_id": env_id,
            "request_id": request_id,
            "attribute": attribute,
            "success": success,
        }
        if error:
            response_payload["error"] = error

        self.bus.publish.remote(
            self.simulator_id,
            "attribute_update_result",
            **response_payload,
        )

    def get_state(self):
        """Get current simulator state (for debugging)."""
        return {
            "initialized": self.initialized,
            "observations": self.current_observations,
            "rewards": self.current_rewards,
            "dones": self.current_dones,
            "info": self.current_info,
        }
