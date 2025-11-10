"""EventBus-backed Simulator proxy.

This module exposes a SimulatorAPI-compatible proxy that communicates with a
remote simulator through the EventBus. It enables the existing
`SumoEnvironment` to interact with SUMO without direct API calls.
"""

# Now, this file is not used to define any Ray actor. It simply defines a class that acts as a proxy to the simulator via the EventBus.

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import ray

from .simulator_api import SimulatorAPI


class EventBusSimulatorProxy(SimulatorAPI):
    """SimulatorAPI implementation that talks to a remote simulator via EventBus."""

    def __init__(
        self,
        bus,
        env_id: str,
        simulator_id: str,
        timeout: float = 10.0,
    ) -> None:
        self.bus = bus
        self.env_id = env_id
        self.simulator_id = simulator_id
        self.timeout = timeout

        # Cached state from most recent responses
        self._agent_ids: List[str] = []
        self._latest_observations: Dict[str, Any] = {}
        self._latest_rewards: Dict[str, float] = {}
        self._latest_dones: Dict[str, bool] = {}
        self._latest_info: Dict[str, Any] = {}
        self._latest_metrics: Dict[str, Any] = {}
        self._latest_system_info: Dict[str, Any] = {}
        self._latest_per_agent_info: Dict[str, Any] = {}

        self._observation_spaces: Dict[str, Any] = {}
        self._action_spaces: Dict[str, Any] = {}

        self._sumo_seed: Optional[Any] = None

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _request(
        self,
        topic: str,
        response_topic: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        required: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Publish a message and wait for the corresponding response."""
        payload = dict(payload or {})
        request_id = str(uuid.uuid4())
        payload.setdefault("env_id", self.env_id)
        payload.setdefault("target_simulator", self.simulator_id)
        payload["request_id"] = request_id

        # Publish request
        ray.get(self.bus.publish.remote(self.env_id, topic, **payload))

        # Wait for response
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            message = ray.get(self.bus.get_message.remote(response_topic, request_id))
            if message:
                if message["sender_id"] != self.simulator_id:
                    # Ignore unexpected senders
                    continue
                return message["payload"]
            time.sleep(0.01)

        if required:
            raise TimeoutError(
                f"EventBus request '{topic}' timed out after {timeout or self.timeout:.2f}s"
            )
        return None

    # ------------------------------------------------------------------
    # SimulatorAPI implementation
    # ------------------------------------------------------------------

    def initialize(self):
        response = self._request("init_request", "init_result")
        if not response or not response.get("success", False):
            raise RuntimeError(response.get("error", "Initialization failed"))

        self._agent_ids = response.get("agent_ids", [])
        self._latest_observations = response.get("observations", {})
        return self._latest_observations

    def reset(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self._sumo_seed is not None:
            payload["seed"] = self._sumo_seed
        response = self._request("reset", "reset_result", payload)
        if not response or not response.get("success", False):
            raise RuntimeError(response.get("error", "Reset failed"))

        self._agent_ids = response.get("agent_ids", self._agent_ids)
        self._latest_observations = response.get("observations", {})
        self._latest_rewards = {aid: 0.0 for aid in self._agent_ids}
        self._latest_dones = {aid: False for aid in self._agent_ids}
        self._latest_dones["__all__"] = False
        self._latest_info = {}
        return self._latest_observations

    def step(
        self, actions: Dict[str, Any]
    ) -> Any:
        response = self._request("action", "step_result", {"actions": actions})
        if not response or not response.get("success", False):
            raise RuntimeError(response.get("error", "Step failed"))

        self._latest_observations = response.get("observations", {})
        self._latest_rewards = response.get("rewards", {})
        self._latest_dones = response.get("dones", {})
        self._latest_info = response.get("info", {})
        self._latest_metrics = response.get("metrics", {})
        self._latest_system_info = response.get("system_info", {})
        self._latest_per_agent_info = response.get("per_agent_info", {})
        return (
            self._latest_observations,
            self._latest_rewards,
            self._latest_dones,
            self._latest_info,
        )

    def close(self):
        response = self._request("close", "close_result", required=False)
        if response and not response.get("success", False):
            raise RuntimeError(response.get("error", "Close failed"))

    def get_agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    def get_observation_space(self, agent_id: str):
        if agent_id not in self._observation_spaces:
            response = self._request(
                "observation_space_request",
                "observation_space_result",
                {"agent_id": agent_id},
            )
            if not response or not response.get("success", False):
                raise RuntimeError(response.get("error", "Observation space unavailable"))
            self._observation_spaces[agent_id] = response.get("space")
        return self._observation_spaces[agent_id]

    def get_action_space(self, agent_id: str):
        if agent_id not in self._action_spaces:
            response = self._request(
                "action_space_request",
                "action_space_result",
                {"agent_id": agent_id},
            )
            if not response or not response.get("success", False):
                raise RuntimeError(response.get("error", "Action space unavailable"))
            self._action_spaces[agent_id] = response.get("space")
        return self._action_spaces[agent_id]

    def get_sim_step(self) -> float:
        if "step" in self._latest_info:
            return self._latest_info["step"]
        response = self._request("sim_step_request", "sim_step_result", required=False)
        if response and response.get("success", False):
            step = response.get("step", 0.0)
            self._latest_info["step"] = step
            return step
        return 0.0

    # ------------------------------------------------------------------
    # Additional helper accessors used by the environment
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        if not self._latest_metrics:
            response = self._request("metrics_request", "metrics_result", required=False)
            if response and response.get("success", False):
                self._latest_metrics = response.get("metrics", {})
        return dict(self._latest_metrics)

    def get_system_info(self) -> Dict[str, Any]:
        if not self._latest_system_info:
            response = self._request("system_info_request", "system_info_result", required=False)
            if response and response.get("success", False):
                self._latest_system_info = response.get("system_info", {})
        return dict(self._latest_system_info)

    def get_per_agent_info(self) -> Dict[str, Any]:
        if not self._latest_per_agent_info:
            response = self._request(
                "per_agent_info_request",
                "per_agent_info_result",
                required=False,
            )
            if response and response.get("success", False):
                self._latest_per_agent_info = response.get("per_agent_info", {})
        return dict(self._latest_per_agent_info)

    def get_rgb_array(self):  # pragma: no cover - rendering is optional
        response = self._request("rgb_array_request", "rgb_array_result", required=False)
        if response and response.get("success", False):
            return response.get("frame")
        return None

    def get_state(self):
        """Get the full state of the remote simulator (for debugging)."""
        response = self._request("state_request", "state_result", required=False)
        if response and response.get("success", False):
            return response.get("state")
        return None

    # ------------------------------------------------------------------
    # SUMO configuration helpers
    # ------------------------------------------------------------------

    @property
    def sumo_seed(self):
        return self._sumo_seed

    @sumo_seed.setter
    def sumo_seed(self, value):
        self._sumo_seed = value
        try:
            self._request(
                "attribute_update",
                "attribute_update_result",
                {"attribute": "sumo_seed", "value": value},
                required=False,
            )
        except TimeoutError:
            # Setting the attribute is best-effort; ignore timeout
            pass