"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation.

This class is now independent of SUMO/traci - it only handles RL logic (observation, reward, action).
All simulation-specific data is provided through a data provider interface.
"""

from typing import Callable, List, Union, Dict, Any

import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: 
    Currently it is not supporting all-red phases (but should be easy to implement it). It's meant that the agent decides only the green phase durations, the yellow time and all-red phases are fixed.

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [ lane_1_density, lane_2_density, ..., lane_n_density,
            lane_1_queue, lane_2_queue, ..., lane_n_queue, 
            lane_1_occupancy, lane_2_occupancy, ..., lane_n_occupancy,
            lane_1_average_speed, lane_2_average_speed, ..., lane_n_average_speed ]

    where:
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
    - ```lane_i_occupancy``` is the sum of lengths of vehicles in incoming lane i divided by the length of the lane
    - ```lane_i_average_speed``` is the average speed of vehicles in incoming lane i divided

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is a continuous vector of size equal to the number of green phases 
    of the traffic signal. Each element of the vector represents the proportion of 
    green time allocated to the corresponding green phase in the next cycle. The 
    actual green time for each phase is computed by normalizing the action vector 
    and scaling it to fit

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        data_provider: Any,  # Interface to get traffic data (replaces direct SUMO access)
        num_green_phases: int,
        observation_class: type,
        detectors: List = None,
    ):
        """Initializes a TrafficSignal object.

        Args:
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            data_provider: Object that provides traffic data (replaces direct SUMO/traci access).
            num_green_phases (int): Number of green phases for this traffic signal.
            observation_class: Class for computing observations.
            detectors (List): List of detector IDs [e1_detectors, e2_detectors].
        """
        self.id = ts_id
        self.data_provider = data_provider  # Replaces self.sumo
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.num_green_phases = num_green_phases
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.detectors = detectors if detectors else [[], []]
        self.avg_veh_length = 3.0
        self.sampling_interval_s = 10
        self.aggregation_interval_s = delta_time

        # Calculate total green and yellow time in a cycle
        self.total_yellow_time = self.yellow_time * self.num_green_phases
        self.total_green_time = self.delta_time - self.total_yellow_time

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = observation_class(self)

        # Get lanes and detectors from data provider
        self.lanes = self.data_provider.get_controlled_lanes(self.id)
        
        self.detectors_e1 = self.detectors[0]
        self.detectors_e2 = self.detectors[1]
        self.detectors_e2_length = {
            e2: self.data_provider.get_detector_length(e2) for e2 in self.detectors_e2
        }

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Box(
            low=np.array([(self.min_green / self.total_green_time) for _ in range(self.num_green_phases)]), 
            high=np.array([1.0 for _ in range(self.num_green_phases)]), 
            dtype=np.float32
        )
        assert (self.min_green * self.num_green_phases) <= self.total_green_time, (
            "Minimum green time too high for traffic signal " + self.id + " cycle time"
        )
        
        # Initialize detector history
        self.detector_history = {
            "density": {det_id: [] for det_id in self.detectors_e2},
            "queue": {det_id: [] for det_id in self.detectors_e2},
            "occupancy": {det_id: [] for det_id in self.detectors_e2},
            "average_speed": {det_id: [] for det_id in self.detectors_e2},
        }
        self._last_sampling_time = -self.sampling_interval_s
        
    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.data_provider.should_act(self.id, self.next_action_time)

    def update(self):
        """Updates the traffic signal state. (No-op since simulator handles the cycle)."""
        pass

    def update_detectors_history(self):
        """
        Cập nhật lịch sử dữ liệu từ các detector mỗi sampling_interval_s (10s).
        
        Luồng hoạt động:
        1. Mỗi 10s, thu thập một mẫu dữ liệu (density, queue, occupancy, speed) từ tất cả detectors
        2. Lưu mẫu vào lịch sử cho từng loại dữ liệu
        3. Giữ lại tối đa max_samples mẫu trong cửa sổ delta_time gần nhất
        4. Khi agent quan sát, các hàm getter sẽ trả về trung bình của các mẫu trong cửa sổ
        """
        current_time = self.data_provider.get_sim_time()
        
        # Kiểm tra xem có cần cập nhật lịch sử không (mỗi sampling_interval_s = 10s)
        if current_time - self._last_sampling_time >= self.sampling_interval_s - 0.1:
            self._last_sampling_time = current_time
            
            # Số lượng mẫu tối đa để giữ trong cửa sổ delta_time
            max_samples = max(1, int(self.aggregation_interval_s / self.sampling_interval_s))
            
            for det_id in self.detectors_e2:
                # --- THU THẬP DENSITY ---
                density = self._compute_detector_density(det_id)
                self.detector_history["density"][det_id].append(density)
                if len(self.detector_history["density"][det_id]) > max_samples:
                    self.detector_history["density"][det_id] = self.detector_history["density"][det_id][-max_samples:]
                
                # --- THU THẬP QUEUE ---
                queue = self._compute_detector_queue(det_id)
                self.detector_history["queue"][det_id].append(queue)
                if len(self.detector_history["queue"][det_id]) > max_samples:
                    self.detector_history["queue"][det_id] = self.detector_history["queue"][det_id][-max_samples:]
                
                # --- THU THẬP OCCUPANCY ---
                occupancy = self._compute_detector_occupancy(det_id)
                self.detector_history["occupancy"][det_id].append(occupancy)
                if len(self.detector_history["occupancy"][det_id]) > max_samples:
                    self.detector_history["occupancy"][det_id] = self.detector_history["occupancy"][det_id][-max_samples:]
                
                # --- THU THẬP AVERAGE SPEED ---
                avg_speed = self._compute_detector_average_speed(det_id)
                self.detector_history["average_speed"][det_id].append(avg_speed)
                if len(self.detector_history["average_speed"][det_id]) > max_samples:
                    self.detector_history["average_speed"][det_id] = self.detector_history["average_speed"][det_id][-max_samples:]
    
    def _compute_detector_density(self, det_id: str) -> float:
        """Tính mật độ giao thông chuẩn hóa [0,1] cho một detector."""
        try:
            vehicle_count = self.data_provider.get_detector_vehicle_count(det_id)
            
            if vehicle_count == 0:
                return 0.0
            
            detector_length_meters = self.data_provider.get_detector_length(det_id)
            if detector_length_meters <= 0:
                return 0.0
            
            vehicle_ids = self.data_provider.get_detector_vehicle_ids(det_id)
            
            if len(vehicle_ids) > 0:
                total_length = sum(self.data_provider.get_vehicle_length(veh_id) for veh_id in vehicle_ids)
                avg_vehicle_length = total_length / len(vehicle_ids)
            else:
                avg_vehicle_length = 5.0
            
            max_vehicle_capacity = detector_length_meters / (self.MIN_GAP + avg_vehicle_length)
            density = vehicle_count / max_vehicle_capacity
            
            return min(1.0, density)
        except Exception:
            return 0.0
    
    def _compute_detector_queue(self, det_id: str) -> float:
        """Tính độ dài hàng đợi chuẩn hóa [0,1] cho một detector."""
        try:
            queue_length_meters = self.data_provider.get_detector_jam_length(det_id)
            
            if queue_length_meters == 0:
                return 0.0
            
            detector_length_meters = self.data_provider.get_detector_length(det_id)
            if detector_length_meters <= 0:
                return 0.0
            
            normalized_queue = queue_length_meters / detector_length_meters
            return min(1.0, normalized_queue)
        except Exception:
            return 0.0
    
    def _compute_detector_occupancy(self, det_id: str) -> float:
        """Tính độ chiếm dụng chuẩn hóa [0,1] cho một detector."""
        try:
            occupancy = self.data_provider.get_detector_occupancy(det_id)
            normalized_occupancy = occupancy / 100.0
            return min(1.0, max(0.0, normalized_occupancy))
        except Exception:
            return 0.0
    
    def _compute_detector_average_speed(self, det_id: str) -> float:
        """Tính tốc độ trung bình chuẩn hóa [0,1] cho một detector."""
        try:
            mean_speed = self.data_provider.get_detector_mean_speed(det_id)
            
            if mean_speed <= 0:
                return 0.0
            
            lane_id = self.data_provider.get_detector_lane_id(det_id)
            max_speed = self.data_provider.get_lane_max_speed(lane_id)
            
            if max_speed > 0:
                normalized_speed = mean_speed / max_speed
                return min(1.0, normalized_speed)
            else:
                return 1.0
        except Exception:
            return 1.0

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase.

        Args:
            new_phase (int): Action representing green time ratios for each phase.
        """
        self.green_times = self._get_green_time_from_ratio(new_phase)

        # Delegate phase setting to data provider (simulator)
        self.data_provider.set_traffic_light_phase(self.id, self.green_times)

        # Set the next action time
        current_time = self.data_provider.get_sim_time()
        self.next_action_time = current_time + self.delta_time

    def _get_green_time_from_ratio(self, green_time_set: np.ndarray):
        """
        Computes the green time for each phase based on the provided green time rates.
        Args:
            green_time_set (np.ndarray): An array of green time rates for each phase.
        Returns:
            List[int]: A list of green times for each phase.
        """
        if np.sum(green_time_set) == 0:
            green_time_set = np.ones(self.num_green_phases)
        green_time_set /= np.sum(green_time_set)
        green_times = green_time_set * self.total_green_time
        
        return green_times

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
        else:
            self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards

        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def _observation_fn_default(self):
        """Default observation function returning comprehensive traffic state information.
        
        Returns:
            np.ndarray: Observation vector containing:
                - phase_id: one-hot encoding of current green phase
                - min_green: whether minimum green time has passed
                - density: normalized vehicle density per incoming lane
                - queue: normalized queue length per incoming lane  
                - occupancy: normalized lane occupancy per incoming lane
                - average_speed: normalized average speed per incoming lane
        """
        # Traffic state information
        density = self.get_lanes_density_by_detectors()
        queue = self.get_lanes_queue_by_detectors()
        occupancy = self.get_lanes_occupancy_by_detectors()
        average_speed = self.get_lanes_average_speed_by_detectors()
        
        # Combine all information
        observation = np.array(density + queue + occupancy + average_speed, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.data_provider.get_lane_vehicles(lane)
            wait_time = 0.0
            for veh in veh_list:
                wait_time += self.data_provider.get_vehicle_waiting_time(veh, lane)
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            speed = self.data_provider.get_vehicle_speed(v)
            allowed_speed = self.data_provider.get_vehicle_allowed_speed(v)
            avg_speed += speed / allowed_speed if allowed_speed > 0 else 0.0
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        # This would need out_lanes from data provider
        return 0.0  # Placeholder - needs out_lanes implementation

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        total_queued = 0
        for lane in self.lanes:
            total_queued += self.data_provider.get_lane_halting_number(lane)
        return total_queued

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.data_provider.get_lane_vehicles(lane)
        return veh_list

    def get_lanes_density_by_detectors(self) -> List[float]:
        """Trả về mật độ trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa mật độ chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_densities = []
        for det_id in self.detectors_e2:
            history = self.detector_history["density"].get(det_id, [])
            if history:
                avg_densities.append(float(np.mean(history)))
            else:
                avg_densities.append(0.0)
        return avg_densities

    def get_lanes_queue_by_detectors(self) -> List[float]:
        """Trả về hàng đợi trung bình trong khoảng delta_time cho mỗi detector.

        Returns:
            List[float]: Danh sách chứa độ dài hàng đợi chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_queues = []
        for det_id in self.detectors_e2:
            history = self.detector_history["queue"].get(det_id, [])
            if history:
                avg_queues.append(float(np.mean(history)))
            else:
                avg_queues.append(0.0)
        return avg_queues

    def get_lanes_occupancy_by_detectors(self) -> List[float]:
        """Trả về độ chiếm dụng trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa độ chiếm dụng chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_occupancies = []
        for det_id in self.detectors_e2:
            history = self.detector_history["occupancy"].get(det_id, [])
            if history:
                avg_occupancies.append(float(np.mean(history)))
            else:
                avg_occupancies.append(0.0)
        return avg_occupancies
    
    def get_lanes_average_speed_by_detectors(self) -> List[float]:
        """Trả về tốc độ trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa tốc độ trung bình chuẩn hóa [0,1] cho mỗi detector.
        """
        avg_speeds = []
        for det_id in self.detectors_e2:
            history = self.detector_history["average_speed"].get(det_id, [])
            if history:
                avg_speeds.append(float(np.mean(history)))
            else:
                avg_speeds.append(1.0)
        return avg_speeds

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }