"""SUMO Simulator Implementation - Complete SUMO backend for traffic simulation.

This module implements the SimulatorAPI for SUMO. It encapsulates all SUMO-specific
logic (traffic signal management, observation, reward, etc.) and provides a clean API
that the environment can use without knowing about SUMO internals.
"""

import os
import sys
import sumolib
import traci
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pyvirtualdisplay.smartdisplay import SmartDisplay

# Add SUMO tools to path
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

from gymnasium import spaces
from .simulator_api import SimulatorAPI

# Import traffic signal and observation functions (still use these internally)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment', 'drl_algo'))
try:
    from environment.drl_algo.traffic_signal import TrafficSignal
    from environment.drl_algo.observations import DefaultObservationFunction
except ImportError:
    # Placeholder if imports fail
    TrafficSignal = None
    DefaultObservationFunction = None

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class SumoSimulator(SimulatorAPI):
    """SUMO traffic simulator implementing SimulatorAPI interface.
    
    This class contains ALL SUMO-related logic:
    - SUMO connection management (start/stop/step)
    - Traffic signal creation and control
    - Observation and reward computation
    - State management
    
    Note: The environment (env.py) should NEVER import traci or sumolib directly.
    It only uses this class through SimulatorAPI methods.
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        label: str = "0",
        use_gui: bool = False,
        virtual_display: Optional[Tuple[int, int]] = None,
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        enforce_max_green: bool = False,
        reward_fn: Union[str, Dict] = "diff-waiting-time",
        reward_weights: Optional[List[float]] = None,
        observation_class = None,
        sumo_seed: Union[str, int] = "random",
        ts_ids: Optional[List[str]] = None,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
    ):
        """Initialize SUMO simulator with all parameters."""
        # Configuration
        self.net_file = net_file
        self.route_file = route_file
        self.label = label
        self.use_gui = use_gui
        self.virtual_display = virtual_display
        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.observation_class = observation_class or DefaultObservationFunction
        self.sumo_seed = sumo_seed
        self.ts_ids = ts_ids
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd or ""
        
        # State
        self.sumo = None
        self.conn = None
        self.disp = None
        self._started = False
        self.traffic_signals = {}
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

    # =====================================================================
    # SimulatorAPI Implementation
    # =====================================================================

    def initialize(self) -> Dict[str, Any]:
        """Initialize simulator and get initial observations.
        
        Steps:
        1. Start temporary SUMO connection to read network metadata
        2. Get agent IDs (traffic light IDs)
        3. Build TrafficSignal objects for each agent
        4. Start full SUMO simulation
        5. Return initial observations for all agents
        """
        # Start temp connection to read metadata
        self._start_temp_connection()
        
        # Get agent IDs if not provided
        if self.ts_ids is None:
            self.ts_ids = list(self.conn.trafficlight.getIDList())
        
        # Build traffic signals
        self._build_traffic_signals(self.conn)
        
        # Close temp connection
        self._close_connection()
        
        # Start full simulation
        self._start_full_connection()
        
        # Initialize vehicles dict and counters
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0
        
        # Get initial observations
        initial_obs = {}
        for ts_id in self.ts_ids:
            if ts_id in self.traffic_signals:
                initial_obs[ts_id] = self.traffic_signals[ts_id].compute_observation()
        
        return initial_obs

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """Execute one simulation step.
        
        Steps:
        1. Apply actions to traffic signals
        2. Run SUMO simulation until next decision time
        3. Update detector history for all agents
        4. Collect observations, rewards, dones
        5. Return tuple of (obs, rewards, dones, info)
        """
        if self.sumo is None:
            raise RuntimeError("Simulator not initialized. Call initialize() or reset() first.")
        
        # Apply actions for agents that are ready to act
        for ts_id, action in actions.items():
            if ts_id in self.traffic_signals and self.traffic_signals[ts_id].time_to_act:
                self.traffic_signals[ts_id].set_next_phase(action)
        
        # Run simulation until next decision time
        time_to_act = False
        while not time_to_act:
            # Step SUMO simulation
            self.sumo.simulationStep()
            
            # Update vehicle counters
            self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
            self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
            self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()
            
            # Update detector history for all traffic signals
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    self.traffic_signals[ts_id].update_detectors_history()
            
            # Check if any agent can act
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    if self.traffic_signals[ts_id].time_to_act or self.fixed_ts:
                        time_to_act = True
                        break
        
        # Collect observations and rewards for agents that acted
        observations = {}
        rewards = {}
        
        for ts_id in self.ts_ids:
            if ts_id in self.traffic_signals:
                if self.traffic_signals[ts_id].time_to_act or self.fixed_ts:
                    observations[ts_id] = self.traffic_signals[ts_id].compute_observation()
                    rewards[ts_id] = self.traffic_signals[ts_id].compute_reward()
        
        # Determine dones
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.get_sim_step() >= self.sim_max_time
        
        # Prepare info dict
        info = {
            "step": self.get_sim_step(),
            "num_arrived": self.num_arrived_vehicles,
            "num_departed": self.num_departed_vehicles,
            "num_teleported": self.num_teleported_vehicles,
        }
        
        return observations, rewards, dones, info

    def reset(self) -> Dict[str, Any]:
        """Reset simulator to initial state.
        
        Steps:
        1. Close current simulation if running
        2. Clear traffic signals and state
        3. Call initialize() to start fresh
        """
        # Close current simulation if running
        if self._started:
            self._close_connection()
        
        # Reset state
        self.traffic_signals = {}
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0
        
        # Initialize fresh simulation
        return self.initialize()

    def close(self):
        """Clean up and close simulator.
        
        Steps:
        1. Close SUMO connection
        2. Stop virtual display if running
        3. Clean up resources
        """
        self._close_connection()
        self.traffic_signals = {}
        self.vehicles = {}

    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs (traffic signal IDs)."""
        return self.ts_ids or []

    def get_observation_space(self, agent_id: str):
        """Get observation space for agent."""
        if agent_id not in self.traffic_signals:
            return None
        return self.traffic_signals[agent_id].observation_space

    def get_action_space(self, agent_id: str):
        """Get action space for agent."""
        if agent_id not in self.traffic_signals:
            return None
        return self.traffic_signals[agent_id].action_space

    def get_sim_step(self) -> float:
        """Get current simulation time."""
        if self.sumo is None:
            return 0.0
        try:
            return float(self.sumo.simulation.getTime())
        except Exception:
            return 0.0

    @property
    def sim_step(self) -> float:
        """Property for current simulation step."""
        return self.get_sim_step()

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate counters for the current episode."""
        return {
            "num_arrived": self.num_arrived_vehicles,
            "num_departed": self.num_departed_vehicles,
            "num_teleported": self.num_teleported_vehicles,
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Return system-level information for the current state."""
        if self.sumo is None:
            return {}

        try:
            vehicles = list(self.sumo.vehicle.getIDList())
            speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
            waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
            num_backlogged = self.sumo.simulation.getPendingVehiclesNumber()

            return {
                "system_total_running": len(vehicles),
                "system_total_backlogged": num_backlogged,
                "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
                "system_total_arrived": self.num_arrived_vehicles,
                "system_total_departed": self.num_departed_vehicles,
                "system_total_teleported": self.num_teleported_vehicles,
                "system_total_waiting_time": float(sum(waiting_times)),
                "system_mean_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
                "system_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            }
        except Exception:
            return {}

    def get_per_agent_info(self) -> Dict[str, Any]:
        """Return per-agent (traffic signal) information."""
        if not self.traffic_signals or not self.ts_ids:
            return {}

        info: Dict[str, Any] = {}
        try:
            stopped: List[int] = []
            accumulated_waiting: List[float] = []
            average_speed: List[float] = []

            for ts in self.ts_ids:
                signal = self.traffic_signals.get(ts)
                if signal is None:
                    continue

                stopped.append(signal.get_total_queued())
                accumulated_waiting.append(sum(signal.get_accumulated_waiting_time_per_lane()))
                average_speed.append(signal.get_average_speed())

                info[f"{ts}_stopped"] = stopped[-1]
                info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting[-1]
                info[f"{ts}_average_speed"] = average_speed[-1]

            info["agents_total_stopped"] = sum(stopped)
            info["agents_total_accumulated_waiting_time"] = float(sum(accumulated_waiting))
        except Exception:
            return {}

        return info

    def get_rgb_array(self):  # pragma: no cover - depends on GUI availability
        """Return an RGB array representation of the current frame if available."""
        if not self.use_gui or self.disp is None:
            return None

        try:
            image = self.disp.grab()
            if image is None:
                return None
            return np.array(image)
        except Exception:
            return None

    # =====================================================================
    # Internal SUMO Management (Private Methods)
    # =====================================================================

    def _binary(self, gui: bool) -> str:
        """Get SUMO binary path."""
        return sumolib.checkBinary("sumo-gui" if gui else "sumo")

    def _build_cmd(self, gui: bool = False, net_only: bool = False) -> List[str]:
        """Build SUMO command line."""
        binp = self._binary(gui)
        cmd = [binp, "-n", self.net_file]
        
        if not net_only:
            cmd.extend(["-r", self.route_file])
            cmd.extend(["--max-depart-delay", str(self.max_depart_delay)])
            cmd.extend(["--waiting-time-memory", str(self.waiting_time_memory)])
            cmd.extend(["--time-to-teleport", str(self.time_to_teleport)])
        
        if self.begin_time > 0:
            cmd.extend(["-b", str(self.begin_time)])
        
        if self.sumo_seed == "random":
            cmd.append("--random")
        elif self.sumo_seed is not None:
            cmd.extend(["--seed", str(self.sumo_seed)])
        
        if not self.sumo_warnings:
            cmd.append("--no-warnings")
        
        if self.additional_sumo_cmd:
            cmd.extend(self.additional_sumo_cmd.split())
        
        if gui:
            cmd.extend(["--start", "--quit-on-end"])
        
        return cmd

    def _start_temp_connection(self):
        """Start temporary SUMO connection to read network metadata."""
        cmd = self._build_cmd(gui=False, net_only=True)
        
        if LIBSUMO:
            traci.start(cmd)
            self.conn = traci
        else:
            label = f"init_connection_{self.label}"
            traci.start(cmd, label=label)
            self.conn = traci.getConnection(label)
        
        return self.conn

    def _start_full_connection(self):
        """Start full SUMO simulation connection."""
        cmd = self._build_cmd(gui=self.use_gui, net_only=False)
        
        # Setup virtual display if needed
        if self.use_gui and self.virtual_display and not LIBSUMO:
            self.disp = SmartDisplay(size=self.virtual_display)
            self.disp.start()
        
        if LIBSUMO:
            traci.start(cmd)
            self.conn = traci
            self.sumo = traci
        else:
            traci.start(cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
            self.sumo = self.conn
        
        self._started = True
        
        # Setup GUI display
        if self.use_gui or self.virtual_display:
            try:
                if "DEFAULT_VIEW" not in dir(traci.gui):
                    traci.gui.DEFAULT_VIEW = "View #0"
                self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            except Exception:
                pass

    def _close_connection(self):
        """Close SUMO connection and cleanup."""
        if not self._started and self.conn is None:
            return
        
        try:
            if not LIBSUMO and self.conn:
                traci.switch(self.label)
            traci.close()
        finally:
            if self.disp is not None:
                try:
                    self.disp.stop()
                except Exception:
                    pass
                self.disp = None
            self.conn = None
            self.sumo = None
            self._started = False

    def _build_traffic_signals(self, conn):
        """Build TrafficSignal objects for each traffic light.
        
        Args:
            conn: SUMO TraCI connection (temporary or full)
        
        Steps:
        1. Get list of all traffic light IDs from SUMO
        2. For each traffic light:
           - Create TrafficSignal object with data_provider interface
           - Setup observation and action spaces
           - Initialize detector connections
        3. Store in self.traffic_signals dict
        """
        self.traffic_signals = {}
        
        # Get all traffic light IDs from SUMO
        ts_ids = conn.trafficlight.getIDList()
        if not ts_ids:
            self.traffic_signals = {}
            return
        
        for ts_id in ts_ids:
            try:
                # Get number of green phases
                logic = conn.trafficlight.getAllProgramLogics(ts_id)[0]
                num_green_phases = len(logic.phases) // 2
                
                # Get detectors for this traffic signal (placeholder - needs implementation)
                detectors = [[], []]  # [e1_detectors, e2_detectors]
                
                # Create TrafficSignal object with data_provider (self)
                ts = TrafficSignal(
                    ts_id=ts_id,
                    delta_time=self.delta_time,
                    yellow_time=self.yellow_time,
                    min_green=self.min_green,
                    max_green=self.max_green,
                    enforce_max_green=self.enforce_max_green,
                    begin_time=self.begin_time,
                    reward_fn=self.reward_fn.get(ts_id, "diff-waiting-time") if isinstance(self.reward_fn, dict) else self.reward_fn,
                    reward_weights=self.reward_weights,
                    data_provider=self,  # Pass self as data provider
                    num_green_phases=num_green_phases,
                    observation_class=self.observation_class,
                    detectors=detectors,
                )
                
                self.traffic_signals[ts_id] = ts
                
            except Exception as e:
                print(f"Warning: Failed to create TrafficSignal for {ts_id}: {e}")
                continue

    # =====================================================================
    # Data Provider Interface Implementation
    # These methods provide traffic data to TrafficSignal agents
    # =====================================================================

    def get_sim_time(self) -> float:
        """Return current simulation time in seconds."""
        if self.sumo is None:
            return 0.0
        try:
            return float(self.sumo.simulation.getTime())
        except Exception:
            return 0.0

    def should_act(self, ts_id: str, next_action_time: float) -> bool:
        """Check if traffic signal should act at current time."""
        return next_action_time == self.get_sim_time()

    def set_traffic_light_phase(self, ts_id: str, green_times: List[float]):
        """Set traffic light phase durations."""
        try:
            logic = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0]
            num_phases = len(green_times)
            for i in range(num_phases):
                logic.phases[2 * i].duration = green_times[i]
            self.sumo.trafficlight.setProgramLogic(ts_id, logic)
        except Exception as e:
            print(f"Warning: Failed to set traffic light phase for {ts_id}: {e}")

    def get_controlled_lanes(self, ts_id: str) -> List[str]:
        """Get list of lanes controlled by traffic signal."""
        try:
            return list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(ts_id)))
        except Exception:
            return []

    # Detector methods
    def get_detector_length(self, detector_id: str) -> float:
        """Get detector length in meters."""
        try:
            return self.sumo.lanearea.getLength(detector_id)
        except Exception:
            return 0.0

    def get_detector_vehicle_count(self, detector_id: str) -> int:
        """Get number of vehicles in detector."""
        try:
            return self.sumo.lanearea.getLastIntervalVehicleNumber(detector_id)
        except Exception:
            return 0

    def get_detector_vehicle_ids(self, detector_id: str) -> List[str]:
        """Get IDs of vehicles in detector."""
        try:
            return self.sumo.lanearea.getLastIntervalVehicleIDs(detector_id)
        except Exception:
            return []

    def get_detector_jam_length(self, detector_id: str) -> float:
        """Get jam length in detector (meters)."""
        try:
            return self.sumo.lanearea.getJamLengthMeters(detector_id)
        except Exception:
            return 0.0

    def get_detector_occupancy(self, detector_id: str) -> float:
        """Get detector occupancy percentage."""
        try:
            return self.sumo.lanearea.getLastIntervalOccupancy(detector_id)
        except Exception:
            return 0.0

    def get_detector_mean_speed(self, detector_id: str) -> float:
        """Get mean speed in detector (m/s)."""
        try:
            return self.sumo.lanearea.getLastIntervalMeanSpeed(detector_id)
        except Exception:
            return 0.0

    def get_detector_lane_id(self, detector_id: str) -> str:
        """Get lane ID associated with detector."""
        try:
            return self.sumo.lanearea.getLaneID(detector_id)
        except Exception:
            return ""

    # Lane methods
    def get_lane_vehicles(self, lane_id: str) -> List[str]:
        """Get list of vehicle IDs in lane."""
        try:
            return self.sumo.lane.getLastStepVehicleIDs(lane_id)
        except Exception:
            return []

    def get_lane_halting_number(self, lane_id: str) -> int:
        """Get number of halting vehicles in lane."""
        try:
            return self.sumo.lane.getLastStepHaltingNumber(lane_id)
        except Exception:
            return 0

    def get_lane_max_speed(self, lane_id: str) -> float:
        """Get maximum allowed speed in lane (m/s)."""
        try:
            return self.sumo.lane.getMaxSpeed(lane_id)
        except Exception:
            return 50.0  # Default speed

    # Vehicle methods
    def get_vehicle_length(self, vehicle_id: str) -> float:
        """Get vehicle length in meters."""
        try:
            return self.sumo.vehicle.getLength(vehicle_id)
        except Exception:
            return 5.0  # Default vehicle length

    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Get vehicle speed (m/s)."""
        try:
            return self.sumo.vehicle.getSpeed(vehicle_id)
        except Exception:
            return 0.0

    def get_vehicle_allowed_speed(self, vehicle_id: str) -> float:
        """Get vehicle's allowed speed (m/s)."""
        try:
            return self.sumo.vehicle.getAllowedSpeed(vehicle_id)
        except Exception:
            return 50.0  # Default speed

    def get_vehicle_waiting_time(self, vehicle_id: str, lane_id: str) -> float:
        """Get vehicle accumulated waiting time in lane."""
        try:
            # Get accumulated waiting time
            acc = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle_id)
            
            # Track per-lane waiting time
            if not hasattr(self, 'vehicle_waiting_times'):
                self.vehicle_waiting_times = {}
            
            veh_lane = self.sumo.vehicle.getLaneID(vehicle_id)
            if vehicle_id not in self.vehicle_waiting_times:
                self.vehicle_waiting_times[vehicle_id] = {veh_lane: acc}
            else:
                self.vehicle_waiting_times[vehicle_id][veh_lane] = acc - sum(
                    [self.vehicle_waiting_times[vehicle_id][l] 
                     for l in self.vehicle_waiting_times[vehicle_id].keys() if l != veh_lane]
                )
            
            return self.vehicle_waiting_times[vehicle_id].get(lane_id, 0.0)
        except Exception:
            return 0.0