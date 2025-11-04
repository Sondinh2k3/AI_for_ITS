"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""

import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
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
        env,
        ts_id: str,
        detectors: List,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo = sumo
        self.detectors = detectors
        self.avg_veh_length = 3.0

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order, nói chung là xóa các phần tử trùng lặp do đặc tính của dict
        # self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        # self.out_lanes = list(set(self.out_lanes))
        # self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.detectors_e1 = self.detectors[0]
        self.detectors_e2 = self.detectors[1]
        self.detectors_e2_length = {
            e2: self.sumo.lanearea.getLength(e2) for e2 in self.detectors_e2
        }

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Box(low=np.array([0.0 for _ in range(self.num_green_phases)]), high=np.array([1.0 for _ in range(self.num_green_phases)]), dtype=np.float32)

    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state. (No-op since SUMO handles the cycle)."""
        pass

    def set_next_phase(self, action: np.ndarray):
        """
        Sets the phase durations for the next cycle based on the action vector.
        Args:
            action (np.ndarray): A vector of values representing the desired green time proportions for each phase.
        """
        # Normalize the action vector to get probabilities
        action = np.abs(action)
        if np.sum(action) == 0:
            action = np.ones(self.num_green_phases)
        action /= np.sum(action)

        # Calculate total green time available in the cycle
        total_yellow_time = self.yellow_time * self.num_green_phases
        total_green_time = self.delta_time - total_yellow_time

        if total_green_time < self.num_green_phases * self.min_green:
            green_times = [self.min_green] * self.num_green_phases
            self.next_action_time = self.env.sim_step + sum(green_times) + total_yellow_time
        else:
            green_times = action * total_green_time
            
            under_min_indices = np.where(green_times < self.min_green)[0]
            over_min_indices = np.where(green_times >= self.min_green)[0]

            if len(under_min_indices) > 0:
                time_needed = np.sum(self.min_green - green_times[under_min_indices])
                time_available = np.sum(green_times[over_min_indices] - self.min_green)
                
                if time_available > 0:
                    reduction_factor = time_needed / time_available
                    if reduction_factor > 1: reduction_factor = 1
                    
                    green_times[under_min_indices] = self.min_green
                    green_times[over_min_indices] -= (green_times[over_min_indices] - self.min_green) * reduction_factor
            
            self.next_action_time = self.env.sim_step + self.delta_time

        new_logic_phases = []
        for i in range(self.num_green_phases):
            green_duration = int(round(green_times[i]))
            if green_duration > 0:
                new_logic_phases.append(self.sumo.trafficlight.Phase(duration=green_duration, state=self.green_phases[i].state))
                
                next_phase_idx = (i + 1) % self.num_green_phases
                yellow_key = (i, next_phase_idx)
                
                if yellow_key in self.yellow_dict:
                    yellow_phase_definition = self.all_phases[self.yellow_dict[yellow_key]]
                    new_logic_phases.append(self.sumo.trafficlight.Phase(duration=self.yellow_time, state=yellow_phase_definition.state))

        logic = self.sumo.trafficlight.getAllProgramLogics(self.id)[0]
        logic.phases = new_logic_phases
        logic.type = 0
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        
        self.green_phase = 0
        self.time_since_last_phase_change = 0
        self.is_yellow = False

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
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
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
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density_by_detectors(self) -> List[float]:
        """Tính toán mật độ giao thông chuẩn hóa [0,1] cho một danh sách các detector khu vực làn (E2).
        Args:
            detectors (List[str]): Danh sách ID của các lane area detector (E2).

        Returns:
            List[float]: Danh sách chứa mật độ chuẩn hóa [0,1] cho mỗi detector.
                        0 = không có xe, 1 = mật độ tối đa (jam density)
        """
        normalized_densities = []
        for det_id in self.detectors_e2:
            # Lấy số lượng xe trên vùng detector
            vehicle_count = self.sumo.lanearea.getLastStepVehicleNumber(det_id)
            
            # Nếu không có xe, mật độ = 0
            if vehicle_count == 0:
                normalized_densities.append(0.0)
                continue
            
            # Lấy chiều dài của detector (tính bằng mét)
            detector_length_meters = self.sumo.lanearea.getLength(det_id)

            if detector_length_meters > 0:
                # Lấy danh sách xe trên detector để tính chiều dài trung bình
                vehicle_ids = self.sumo.lanearea.getLastStepVehicleIDs(det_id)
                
                try:
                    if len(vehicle_ids) > 0:
                        # Tính chiều dài trung bình của xe hiện có trên detector
                        total_length = sum(self.sumo.vehicle.getLength(veh_id) for veh_id in vehicle_ids)
                        avg_vehicle_length = total_length / len(vehicle_ids)
                    else:
                        # Sử dụng chiều dài xe mặc định 5.0m khi không có xe
                        avg_vehicle_length = 5.0
                except Exception:
                    # Fallback nếu có lỗi khi lấy chiều dài xe
                    avg_vehicle_length = 5.0
                
                # Tính mật độ theo công thức tương tự get_lanes_density
                # Số xe tối đa = chiều dài detector / (khoảng cách tối thiểu + chiều dài xe trung bình)
                max_vehicle_capacity = detector_length_meters / (self.MIN_GAP + avg_vehicle_length)
                
                # Tính mật độ chuẩn hóa và giới hạn trong [0,1]
                density = vehicle_count / max_vehicle_capacity
                normalized_density = min(1.0, density)
                    
                normalized_densities.append(normalized_density)
            else:
                # Nếu detector không có chiều dài, mật độ là 0
                normalized_densities.append(0.0)

        return normalized_densities

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_lanes_queue_by_detectors(self) -> List[float]:
        """Tính toán độ dài hàng đợi chuẩn hóa [0,1] cho một danh sách các detector khu vực làn (E2).

        Returns:
            List[float]: Danh sách chứa độ dài hàng đợi chuẩn hóa [0,1] cho mỗi detector.
                        0 = không có xe đang dừng, 1 = hàng đợi tối đa (jam)
        """
        normalized_queues = []
        for det_id in self.detectors_e2:
            # Lấy độ dài xe đang dừng trên vùng detector
            queue_length_meters = self.sumo.lanearea.getJamLengthMeters(det_id)
            
            # Nếu không có xe đang dừng, hàng đợi = 0
            if queue_length_meters == 0:
                normalized_queues.append(0.0)
                continue
            
            # Lấy chiều dài của detector (tính bằng mét)
            detector_length_meters = self.sumo.lanearea.getLength(det_id)

            if detector_length_meters > 0:
                
                # Tính hàng đợi chuẩn hóa
                normalized_queue = queue_length_meters / detector_length_meters
                # Đảm bảo giá trị trong khoảng [0,1]
                normalized_queue = min(1.0, normalized_queue)
                    
                normalized_queues.append(normalized_queue)
            else:
                # Nếu detector không có chiều dài, hàng đợi là 0
                normalized_queues.append(0.0)

        return normalized_queues

    def get_lanes_occupancy(self) -> List[float]:
        """Returns the occupancy [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The occupancy is computed as the sum of the lengths of the vehicles divided by the length of the lane.
        """
        lanes_occupancy = []
        for lane in self.lanes:
            lane_length = self.lanes_length[lane]
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            total_veh_length = sum(self.sumo.vehicle.getLength(veh) for veh in vehs)
            occupancy = total_veh_length / lane_length if lane_length > 0 else 0.0
            lanes_occupancy.append(min(1.0, occupancy))
        return lanes_occupancy

    def get_lanes_occupancy_by_detectors(self) -> List[float]:
        """Tính toán độ chiếm dụng chuẩn hóa [0,1] cho một danh sách các detector khu vực làn (E2).
        
        Args:
            detectors (List[str]): Danh sách ID của các lane area detector (E2).

        Returns:
            List[float]: Danh sách chứa độ chiếm dụng chuẩn hóa [0,1] cho mỗi detector.
                        0 = không có xe chiếm dụng, 1 = chiếm dụng hoàn toàn
        """
        detectors_occupancy = []
        for det_id in self.detectors_e2:
            try:
                # Sử dụng API có sẵn của SUMO để lấy occupancy trực tiếp
                occupancy = self.sumo.lanearea.getLastStepOccupancy(det_id)
                
                # API này trả về giá trị từ 0 đến 100 (phần trăm)
                # Chuẩn hóa về [0,1]
                normalized_occupancy = occupancy / 100.0
                
                # Đảm bảo giá trị trong khoảng [0,1]
                normalized_occupancy = min(1.0, max(0.0, normalized_occupancy))
                
                detectors_occupancy.append(normalized_occupancy)
            
            except Exception:
                # Fallback nếu có lỗi khi gọi API
                detectors_occupancy.append(0.0)
        
        return detectors_occupancy
    
    def get_lanes_average_speed(self) -> List[float]:
        """Returns the average speed [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The average speed is computed as the average speed of the vehicles divided by the maximum allowed speed of the vehicles.
        """
        lanes_average_speed = []
        for lane in self.lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            if len(vehs) == 0:
                lanes_average_speed.append(1.0)
                continue
            total_speed = 0.0
            for veh in vehs:
                speed = self.sumo.vehicle.getSpeed(veh)
                allowed_speed = self.sumo.vehicle.getAllowedSpeed(veh)
                total_speed += speed / allowed_speed if allowed_speed > 0 else 0.0
            avg_speed = total_speed / len(vehs)
            lanes_average_speed.append(min(1.0, avg_speed))
        return lanes_average_speed

    def get_lanes_average_speed_by_detectors(self) -> List[float]:
        """Tính toán tốc độ trung bình chuẩn hóa [0,1] cho một danh sách các detector khu vực làn (E2).
        
        Args:
            detectors (List[str]): Danh sách ID của các lane area detector (E2).

        Returns:
            List[float]: Danh sách chứa tốc độ trung bình chuẩn hóa [0,1] cho mỗi detector.
                        0 = tốc độ = 0, 1 = tốc độ = tốc độ tối đa của làn
                        
        Obs: Sử dụng API getLastStepMeanSpeed của SUMO để lấy tốc độ trung bình trực tiếp.
        """
        detectors_average_speed = []
        for det_id in self.detectors_e2:
            try:
                # Lấy tốc độ trung bình từ API có sẵn (m/s)
                mean_speed = self.sumo.lanearea.getLastStepMeanSpeed(det_id)
                
                # Early return nếu tốc độ = 0
                if mean_speed <= 0:
                    detectors_average_speed.append(0.0)
                    continue
                
                # Lấy tốc độ tối đa cho phép của làn chứa detector
                # Cần lấy lane ID từ detector để biết speed limit
                lane_id = self.sumo.lanearea.getLaneID(det_id)
                max_speed = self.sumo.lane.getMaxSpeed(lane_id)
                
                if max_speed > 0:
                    # Chuẩn hóa tốc độ về [0,1]
                    normalized_speed = mean_speed / max_speed
                    # Đảm bảo giá trị trong khoảng [0,1]
                    normalized_speed = min(1.0, normalized_speed)
                else:
                    # Nếu không có giới hạn tốc độ, trả về 1.0
                    normalized_speed = 1.0
                    
                detectors_average_speed.append(normalized_speed)
            
            except Exception:
                # Fallback nếu có lỗi khi gọi API
                detectors_average_speed.append(1.0)  # Giả định không có tắc nghẽn
        
        return detectors_average_speed

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

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