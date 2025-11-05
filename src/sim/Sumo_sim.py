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
           - Create TrafficSignal object
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
                # Create TrafficSignal object
                ts = TrafficSignal(
                    ts_id,
                    sumo=conn,
                    delta_time=self.delta_time,
                    yellow_time=self.yellow_time,
                    min_green=self.min_green,
                    max_green=self.max_green,
                )
                
                self.traffic_signals[ts_id] = ts
                
            except Exception as e:
                print(f"Warning: Failed to create TrafficSignal for {ts_id}: {e}")
                continue