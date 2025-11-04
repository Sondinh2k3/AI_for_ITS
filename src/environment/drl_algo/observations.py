"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        density = self.ts.get_lanes_density_by_detectors()
        queue = self.ts.get_lanes_queue_by_detectors()
        occupancy = self.ts.get_lanes_occupancy_by_detectors()
        average_speed = self.ts.get_lanes_average_speed_by_detectors()
        observation = np.array(density + queue + occupancy + average_speed, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(4 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(4 * len(self.ts.lanes), dtype=np.float32),
        )