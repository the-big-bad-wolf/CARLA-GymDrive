"""
Reward Function

This is the file where the reward function can be customized. If you need more information than the provided please also change it in the environment.py file.

I made the reward function based on this data:
- FPS: 30
- Ticks/Steps per second: 100
- Episode time: 30 seconds
- Maximum number of ticks/steps: 3000
"""

from calendar import c
from src.carlacore.vehicle import Vehicle
from src.carlacore.world import World
import src.config.configuration as config
import carla
import numpy as np


# ======================================== Global Variables =================================================================
class Reward:
    def __init__(self) -> None:
        self.terminated = False
        self.inside_stop_area = False
        self.has_stopped = False
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.waypoints = []
        self.previous_target_distance = 0.0
        self.total_ep_reward = 0
        self.total_collision_reward = 0
        self.total_steering_jerk_reward = 0
        self.total_throttle_brake_jerk_reward = 0
        self.total_speed_reward = 0
        self.total_steering_reward = 0
        self.total_target_reached_reward = 0
        self.total_target_progress_reward = 0
        self.total_target_progress = 0
        self.total_waypoint_reached_reward = 0

        self.countint = 0

    # ======================================== Main Reward Function ==========================================================
    def calculate_reward(
        self,
        vehicle: Vehicle,
        current_pos,
        target_pos,
        current_waypoint_pos,
        speed,
        min_distance,
        number_of_steps,
    ) -> float:

        if self.terminated:
            self.countint += 1
            print("The episode already ended!!!, count: ", self.countint)

        collision_reward = self.__collision_reward(
            vehicle, min_distance, number_of_steps
        )
        steering_jerk_reward = self.__steering_jerk(vehicle)
        throttle_brake_jerk_reward = self.__throttle_brake_jerk(vehicle)
        speed_reward = self.__speed_reward(speed)
        steering_reward = self.__steering_reward(vehicle)
        target_reached_reward = self.__target_reached(current_pos, target_pos)
        target_progress_reward = self.__target_progress(current_pos, target_pos)
        waypoint_reached_reward = self.__waypoint_reached(
            current_pos, current_waypoint_pos
        )

        total_reward = (
            collision_reward
            + steering_jerk_reward
            + throttle_brake_jerk_reward
            # + speed_reward
            # + steering_reward
            + target_reached_reward
            + target_progress_reward
            + waypoint_reached_reward
        )

        self.total_collision_reward += collision_reward
        self.total_steering_jerk_reward += steering_jerk_reward
        self.total_throttle_brake_jerk_reward += throttle_brake_jerk_reward
        # self.total_speed_reward += speed_reward
        # self.total_steering_reward += steering_reward
        self.total_target_reached_reward += target_reached_reward
        self.total_target_progress_reward += target_progress_reward
        self.total_waypoint_reached_reward += waypoint_reached_reward

        self.total_ep_reward += total_reward
        return total_reward

    # ============================================= Reward Functions ==========================================================
    def __collision_reward(
        self, vehicle: Vehicle, min_distance: float, number_of_steps: int
    ):
        if vehicle.collision_occurred() or min_distance == 0:
            episode_duration = (number_of_steps - 1) / config.ENV_MAX_STEPS
            crash_penalty = -100 + (episode_duration * 100)
            self.terminated = True
            return crash_penalty
        else:
            return -100 / config.ENV_MAX_STEPS

    def __steering_jerk(self, vehicle: Vehicle, threshold=0.0):
        """
        This reward function aims to minimize the sudden changes in the steering value of the vehicle. The reward is calculated as follows:
        {
            0.0     : if the steering value difference is less than the threshold,
            -lambda : if the steering value difference is greater or equal than the threshold
        }

        Based on the calculations, the max reward for this function is 0 and the min reward is -10;
        lambda = 1/300
        """
        steering_diff = (
            abs(vehicle.get_steering() - self.current_steering) ** 2 * config.SIM_FPS
        )
        self.current_steering = vehicle.get_steering()
        return -steering_diff / 800 if steering_diff > threshold else 0.0

    def __throttle_brake_jerk(self, vehicle: Vehicle, threshold=0.1):
        """
        This reward function aims to minimize the sudden changes in the throttle/brake of the vehicle. The reward is calculated as follows:
        {
            0.0     : if the throttle/brake difference is less than the threshold,
            -lambda : if the throttle/brake difference is greater or equal than the threshold
        }

        Based on the calculations, the max reward for this function is 0 and the min reward is -10;
        lambda = 1/300
        """
        throttle_diff = (
            abs(vehicle.get_throttle_brake() - self.current_throttle) ** 2
            * config.SIM_FPS
        )
        self.current_throttle = vehicle.get_throttle_brake()
        return -throttle_diff / 800 if throttle_diff > threshold else 0.0

    def __speed_reward(self, speed, speed_limit=50):
        lbd = 100 / config.ENV_MAX_STEPS
        speed = speed * 3.6  # convert to km/h
        minspeed = 20
        fraction = speed / minspeed
        if fraction < 1:
            return fraction * lbd
        else:
            return lbd

    def __steering_reward(self, vehicle: Vehicle, threshold=0.0):
        """
        This reward function aims to keep the vehicle straight
        """
        steering = vehicle.get_steering()
        return -abs(steering)

    def __target_reached(self, current_pos, target_pos, threshold=5.0):
        target_distance = self.distance(current_pos, target_pos).item()

        if target_distance < threshold:
            self.terminated = True
            return 100.0
        else:
            return 0.0

    def __target_progress(self, current_pos, target_pos, threshold=5.0):
        target_distance = self.distance(current_pos, target_pos).item()
        progress = self.previous_target_distance - target_distance
        self.previous_target_distance = target_distance
        return progress

    def __waypoint_reached(self, current_pos, current_waypoint_pos, threshold=2.0):
        waypoint_distance = self.distance(current_pos, current_waypoint_pos)
        if waypoint_distance < threshold:
            try:
                self.waypoints.pop(0)
            except IndexError:
                pass
            return 10
        else:
            return 0

    def __light_pole_trangression(self, map, vehicle, world):
        """
        This reward function penalizes the agent if it doesn't stop at a stop sign. The reward is calculated as follows:
        {
            0       : if the vehicle stops at the stop sign,
            -lambda : if the vehicle doesn't stop at the stop sign
        }

        Based on precise calculations the max reward for this function is 0 and the min reward is -20.
        """
        lbd = 20.0

        # Get the current waypoint of the vehicle
        current_waypoint = map.get_waypoint(
            vehicle.get_location(), project_to_road=True
        )

        # Get the traffic lights affecting the current waypoint
        traffic_lights = world.get_world().get_traffic_lights_from_waypoint(
            current_waypoint, distance=10.0
        )

        for traffic_light in traffic_lights:
            # Check if the traffic light is red
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # Get the stop waypoints for the traffic light
                stop_waypoints = traffic_light.get_stop_waypoints()

                # Check if the vehicle has passed the stop line
                for stop_waypoint in stop_waypoints:
                    if (
                        current_waypoint.transform.location.distance(
                            stop_waypoint.transform.location
                        )
                        < 2.0
                        and vehicle.get_speed() > 0.3
                    ):
                        self.terminated = True
                        return -lbd

        return 0.0

    def __stop_sign_transgression(self, vehicle, map):
        """
        This reward function penalizes the agent if it doesn't stop at a stop sign. The reward is calculated as follows:
        {
            0       : if the vehicle stops at the stop sign,
            -lambda : if the vehicle doesn't stop at the stop sign
        }

        Based on precise calculations the max reward for this function is 0 and the min reward is -20.
        """
        lbd = 20.0
        distance = 20.0  # meters (adjust as needed)

        current_location = vehicle.get_location()
        current_waypoint = map.get_waypoint(current_location, project_to_road=True)

        # Get all the stop sign landmarks within a certain distance from the vehicle and on the same road
        stop_signs_on_same_road = []
        for landmark in current_waypoint.get_landmarks_of_type(
            distance, carla.LandmarkType.StopSign
        ):
            landmark_waypoint = map.get_waypoint(
                landmark.transform.location, project_to_road=True
            )
            if landmark_waypoint.road_id == current_waypoint.road_id:
                stop_signs_on_same_road.append(landmark)

        if len(stop_signs_on_same_road) == 0:
            if self.inside_stop_area and self.has_stopped:
                print("Vehicle has stopped at the stop sign.")
                self.has_stopped = False
                self.inside_stop_area = False
                return 0
            elif self.inside_stop_area and not self.has_stopped:
                print("Vehicle has not stopped at the stop sign.")
                self.has_stopped = False
                self.inside_stop_area = False
                self.terminated = True
                return -lbd
            else:
                return 0.0
        else:
            self.inside_stop_area = True

        # The vehicle entered the stop sign area
        for stop_sign in stop_signs_on_same_road:
            # Check if the vehicle has stopped
            if vehicle.get_speed() < 1.0:
                self.has_stopped = True

    # ==================================== Helper Functions ================================================================
    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_waypoints(self):
        return self.waypoints

    def reset(self, waypoints, vehicle: Vehicle):
        self.terminated = False
        self.inside_stop_area = False
        self.has_stopped = False
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.waypoints = waypoints
        self.total_ep_reward = 0
        self.total_collision_reward = 0
        self.total_steering_jerk_reward = 0
        self.total_throttle_brake_jerk_reward = 0
        self.total_speed_reward = 0
        self.total_steering_reward = 0
        self.total_target_reached_reward = 0
        self.total_target_progress_reward = 0
        self.total_waypoint_reached_reward = 0
        vehicle_loc = vehicle.get_location()
        self.previous_target_distance = self.distance(
            waypoints[-1], np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        )

    def get_terminated(self):
        return self.terminated

    def get_total_ep_reward(self):
        return self.total_ep_reward

    def get_subrewards(self):
        return {
            "collision_reward": self.total_collision_reward,
            "steering_jerk_reward": self.total_steering_jerk_reward,
            "throttle_brake_jerk_reward": self.total_throttle_brake_jerk_reward,
            "speed_reward": self.total_speed_reward,
            "steering_reward": self.total_steering_reward,
            "target_reached_reward": self.total_target_reached_reward,
            "target_progress_reward": self.total_target_progress_reward,
            "waypoint_reached_reward": self.total_waypoint_reached_reward,
        }
