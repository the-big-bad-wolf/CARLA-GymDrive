"""
Sensors Module:
    It provides classes for each CARLA sensor to attach to the vehicle and listen to the data from the sensor using callbacks.

    Available sensors:
        - RGB Camera
        - LiDAR
        - Radar
        - GNSS
        - IMU
        - Collision
        - Lane Invasion
    
    Future Sensors:
        - Semantic Segmentation Camera
        - Instance Segmentation Camera
        - Depth Camera
        - Lidar Semantic Segmentation
        - Obstacle Detection
        - Optical Flow Camera (AKA: Motion Camera)
"""

import carla
import numpy as np
import math
from PIL import Image
import cv2
import src.config.configuration as configuration


# ====================================== RGB Camera ======================================
class RGB_Camera:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_rgb_camera(world, vehicle, sensor_dict)
        self.__last_data = None
        self.__raw_data = None
        self.__sensor_ready = False
        self.__sensor.listen(lambda data: self.callback(data))

    def attach_rgb_camera(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        # attributes
        sensor_bp.set_attribute("image_size_x", str(sensor_dict["image_size_x"]))
        sensor_bp.set_attribute("image_size_y", str(sensor_dict["image_size_y"]))
        sensor_bp.set_attribute("fov", str(sensor_dict["fov"]))
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        camera_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return camera_sensor

    def callback(self, data):
        global configuration

        # Get the image from the data
        image = Image.frombytes(
            "RGBA", (data.width, data.height), data.raw_data, "raw", "RGBA"
        )

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Take out the alpha channel
        image_array = image_array[:, :, :3]

        self.__raw_data = image_array
        self.__sensor_ready = True

        # Ensure the array is contiguous in memory
        image_array = np.ascontiguousarray(image_array)

        # Display the processed image using Pygame
        self.__last_data = image_array

        # Save image in directory
        if configuration.VERBOSE:
            timestamp = data.timestamp
            cv2.imwrite(f"data/rgb_camera/{timestamp}.png", image_array)

    def get_last_data(self):
        return self.__last_data

    def get_data(self):
        if self.__raw_data is not None:
            return self.__raw_data
        else:
            return np.zeros((480, 640, 3))

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== LiDAR ======================================
class Lidar:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_lidar(world, vehicle, sensor_dict)
        self.__last_data = None
        self.__raw_data = None
        self.__sensor_ready = False
        self.__sensor.listen(lambda data: self.callback(data))

    def attach_lidar(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        # attributes
        sensor_bp.set_attribute("channels", str(sensor_dict["channels"]))
        sensor_bp.set_attribute(
            "points_per_second", str(sensor_dict["points_per_second"])
        )
        sensor_bp.set_attribute(
            "rotation_frequency", str(sensor_dict["rotation_frequency"])
        )
        sensor_bp.set_attribute("range", str(sensor_dict["range"]))
        sensor_bp.set_attribute("upper_fov", str(sensor_dict["upper_fov"]))
        sensor_bp.set_attribute("lower_fov", str(sensor_dict["lower_fov"]))
        sensor_bp.set_attribute("horizontal_fov", str(sensor_dict["horizontal_fov"]))
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        lidar_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return lidar_sensor

    def callback(self, data):
        global configuration

        # Assuming lidar_data is the raw Lidar data
        lidar_data = data.raw_data
        lidar_data = np.frombuffer(lidar_data, dtype=np.dtype("f4"))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Ensure a fixed number of points (e.g., 400)
        fixed_num_points = 500
        if lidar_data.shape[0] < fixed_num_points:
            # Pad with zeros if fewer points than expected
            lidar_data = np.pad(
                lidar_data,
                ((0, fixed_num_points - lidar_data.shape[0]), (0, 0)),
                mode="constant",
            )
        elif lidar_data.shape[0] > fixed_num_points:
            # Downsample if more points than expected
            indices = np.linspace(
                0, lidar_data.shape[0] - 1, fixed_num_points, dtype=int
            )
            lidar_data = lidar_data[indices]

        # Update self.__raw_data with the modified Lidar data
        self.__raw_data = lidar_data
        self.__sensor_ready = True

        # Extract X, Y, Z coordinates and intensity values
        points_xyz = lidar_data[:, :3]
        intensity = lidar_data[:, 3]

        # Intensity scaling factor
        intensity_scale = 10.0  # Adjust this value to control the brightness

        # Create a 2D histogram with a predetermined size
        width, height = 640, 360
        lidar_image_array = np.zeros((height, width))

        # Scale and shift X and Y coordinates to fit within the histogram size
        x_scaled = ((points_xyz[:, 0] + 50) / 100) * (width - 1)
        y_scaled = ((points_xyz[:, 1] + 50) / 100) * (height - 1)

        # Round the scaled coordinates to integers
        x_indices = np.round(x_scaled).astype(int)
        y_indices = np.round(y_scaled).astype(int)

        # Clip the indices to stay within the image bounds
        x_indices = np.clip(x_indices, 0, width - 1)
        y_indices = np.clip(y_indices, 0, height - 1)

        # Assign scaled intensity values to the corresponding pixel in the histogram
        lidar_image_array[y_indices, x_indices] = intensity * intensity_scale

        # Clip the intensity values to stay within the valid color range
        lidar_image_array = np.clip(lidar_image_array, 0, 255)

        # Display the processed image using Pygame
        self.__last_data = lidar_image_array

        # Save image in directory
        if configuration.VERBOSE:
            timestamp = data.timestamp
            cv2.imwrite(f"data/lidar/{timestamp}.png", lidar_image_array)

    def get_last_data(self):
        return self.__last_data

    def get_data(self):
        return self.__raw_data

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== Radar ======================================
class Radar:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_radar(world, vehicle, sensor_dict)
        self.__last_data = None
        self.__raw_data = None
        self.__sensor_ready = False
        self.__sensor.listen(lambda data: self.callback(data))

    def attach_radar(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.other.radar")
        # attributes
        sensor_bp.set_attribute("horizontal_fov", str(sensor_dict["horizontal_fov"]))
        sensor_bp.set_attribute("vertical_fov", str(sensor_dict["vertical_fov"]))
        sensor_bp.set_attribute(
            "points_per_second", str(sensor_dict["points_per_second"])
        )
        sensor_bp.set_attribute("range", str(sensor_dict["range"]))
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        radar_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return radar_sensor

    def callback(self, data):
        global configuration

        # Get the radar data
        radar_data = data.raw_data

        points = np.frombuffer(radar_data, dtype=np.dtype("f4"))
        self.__raw_data = points
        self.__sensor_ready = True
        points = np.reshape(points, (len(data), 4))

        # Extract information from radar points
        azimuths = points[:, 1]
        depths = points[:, 3]

        # Create a 2D histogram with a predetermined size
        width, height = 640, 360
        radar_image_array = np.zeros((height, width))

        # Scale azimuth values to fit within the histogram size
        azimuth_scaled = ((np.degrees(azimuths) + 180) / 360) * (width - 1)

        # Scale depth values to fit within the histogram size
        depth_scaled = (depths / 100) * (height - 1)

        # Round the scaled azimuth and depth values to integers
        azimuth_indices = np.round(azimuth_scaled).astype(int)
        depth_indices = np.round(depth_scaled).astype(int)

        # Clip the indices to stay within the image bounds
        azimuth_indices = np.clip(azimuth_indices, 0, width - 1)
        depth_indices = np.clip(depth_indices, 0, height - 1)

        # Set a value (e.g., velocity) at each (azimuth, depth) coordinate in the histogram
        radar_image_array[depth_indices, azimuth_indices] = (
            255  # Set a constant value for visibility
        )

        self.__last_data = radar_image_array

        # Save image in directory
        if configuration.VERBOSE:
            timestamp = data.timestamp
            cv2.imwrite(f"data/radar/{timestamp}.png", radar_image_array)

    def get_last_data(self):
        return self.__last_data

    def get_data(self):
        return self.__raw_data

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== GNSS ======================================
class GNSS:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_gnss(world, vehicle, sensor_dict)
        self.__last_data = None
        self.__sensor_ready = False
        self.__sensor.listen(lambda data: self.callback(data))

    def attach_gnss(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.other.gnss")
        # attributes
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        gnss_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return gnss_sensor

    def callback(self, data):
        global configuration
        self.__last_data = data
        self.__sensor_ready = True

    def get_last_data(self):
        return self.__last_data

    def get_data(self):
        return np.array(
            [
                self.__last_data.latitude,
                self.__last_data.longitude,
                self.__last_data.altitude,
            ]
        )

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== IMU ======================================
class IMU:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_imu(world, vehicle, sensor_dict)
        self.__sensor.listen(lambda data: self.callback(data))
        self.__sensor_ready = False

    def attach_imu(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.other.imu")
        # attributes
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        imu_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return imu_sensor

    def callback(self, data):
        global configuration
        self.__last_data = data
        self.__sensor_ready = True

    def get_last_data(self):
        return self.__last_data

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== Collision ======================================
class Collision:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_collision(world, vehicle, sensor_dict)
        self.__sensor.listen(lambda data: self.callback(data))
        self.__sensor_ready = True
        self.critical_collision = False

    def attach_collision(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.other.collision")

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        collision_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return collision_sensor

    def callback(self, data):
        if configuration.VERBOSE:
            print(f"Collision Occurred at {data.timestamp} with {data.other_actor}")
        self.critical_collision = True

    def collision_occurred(self):
        return self.critical_collision

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


# ====================================== Lane Invasion ======================================
class Lane_Invasion:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_lane_invasion(world, vehicle, sensor_dict)
        self.__sensor.listen(lambda data: self.callback(data))
        self.__sensor_ready = True
        self.lane_transgression = False

    def attach_lane_invasion(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            )
        )
        lane_invasion_sensor = world.spawn_actor(
            sensor_bp, transform, attach_to=vehicle
        )

        return lane_invasion_sensor

    def callback(self, data):
        for lane_marking in data.crossed_lane_markings:
            if (
                lane_marking.type == carla.LaneMarkingType.NONE
                or lane_marking.type == carla.LaneMarkingType.Grass
                or lane_marking.type == carla.LaneMarkingType.Curb
            ):
                self.lane_transgression = True
                break

        if configuration.VERBOSE:
            for marking in data.crossed_lane_markings:
                print(f"Lane Invasion Occurred at {data.timestamp} with {marking.type}")

    def is_ready(self):
        return self.__sensor_ready

    def lane_invasion_occurred(self):
        return self.lane_transgression

    def destroy(self):
        self.__sensor.destroy()


class BEV_Camera:
    def __init__(self, world, vehicle, sensor_dict):
        self.__sensor = self.attach_bev_camera(world, vehicle, sensor_dict)
        self.__last_data = None
        self.__raw_data = None
        self.__sensor_ready = False
        self.__sensor.listen(lambda data: self.callback(data))

    def attach_bev_camera(self, world, vehicle, sensor_dict):
        sensor_bp = world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        # attributes
        sensor_bp.set_attribute("image_size_x", str(sensor_dict["image_size_x"]))
        sensor_bp.set_attribute("image_size_y", str(sensor_dict["image_size_y"]))
        sensor_bp.set_attribute("fov", str(sensor_dict["fov"]))
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))

        # This will place the camera in the front bumper of the car
        transform = carla.Transform(
            carla.Location(
                x=sensor_dict["location_x"],
                y=sensor_dict["location_y"],
                z=sensor_dict["location_z"],
            ),
            carla.Rotation(
                pitch=-90.0,
                yaw=0.0,
                roll=0.0,
            ),
        )
        camera_sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        return camera_sensor

    def callback(self, data):
        global configuration

        data.convert(carla.ColorConverter.CityScapesPalette)
        # Get the image from the data
        image = Image.frombytes(
            "RGBA", (data.width, data.height), data.raw_data, "raw", "RGBA"
        )

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Take out the alpha channel
        image_array = image_array[:, :, :3]

        self.__raw_data = image_array
        self.__sensor_ready = True

        # Ensure the array is contiguous in memory
        image_array = np.ascontiguousarray(image_array)

        # Display the processed image using Pygame
        self.__last_data = image_array

        # Save image in directory
        if configuration.VERBOSE:
            timestamp = data.timestamp
            cv2.imwrite(f"data/rgb_camera/{timestamp}.png", image_array)

    def get_last_data(self):
        return self.__last_data

    def get_data(self):
        if self.__raw_data is not None:
            return self.__raw_data
        else:
            return np.zeros((480, 640, 3))

    def is_ready(self):
        return self.__sensor_ready

    def destroy(self):
        self.__sensor.destroy()


class Circogram:
    def __init__(self, world, vehicle, sensor_dict):
        self.__world = world
        self.__sensors = self.attach_circogram(world, vehicle, sensor_dict)
        self.__sensor_ready_list = [False] * len(self.__sensors)
        self.__circogram: list[None | float] = [None] * len(self.__sensors)
        self.__circogram_velocity_x = [0.0] * len(self.__sensors)
        self.__circogram_velocity_y = [0.0] * len(self.__sensors)
        self.__drivable_tags = {1: "Roads", 24: "RoadLine"}
        self.__ego_idx = vehicle.id
        for sensor, angle in self.__sensors:
            sensor.listen(
                lambda point_cloud, angle=angle: self.callback(point_cloud, angle)
            )

    def attach_circogram(self, world, vehicle, sensor_dict) -> list[tuple]:
        vehicle_width = vehicle.bounding_box.extent.x * 2
        vehicle_length = vehicle.bounding_box.extent.y * 2
        vehicle_height = vehicle.bounding_box.extent.z * 2

        num_rays = sensor_dict["num_rays"]
        angles = [i * 360 / num_rays for i in range(num_rays)]
        sensor_locations = []
        for angle in angles:
            rad = math.radians(angle)
            if math.cos(rad) != 0:
                slope = math.sin(rad) / math.cos(rad)
                if abs(slope) <= vehicle_length / vehicle_width:
                    x = vehicle_width / 2 if math.cos(rad) > 0 else -vehicle_width / 2
                    y = slope * x
                else:
                    y = vehicle_length / 2 if math.sin(rad) > 0 else -vehicle_length / 2
                    x = y / slope
            else:
                x = 0
                y = vehicle_length / 2 if math.sin(rad) > 0 else -vehicle_length / 2

            sensor_locations.append(((x, y), angle))

        sensors = []
        sensor_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        sensor_bp.set_attribute("channels", str(sensor_dict["channels"]))
        sensor_bp.set_attribute(
            "points_per_second", str(sensor_dict["points_per_second"])
        )
        sensor_bp.set_attribute("range", str(sensor_dict["range"]))
        sensor_bp.set_attribute("upper_fov", str(sensor_dict["upper_fov"]))
        sensor_bp.set_attribute("lower_fov", str(sensor_dict["lower_fov"]))
        sensor_bp.set_attribute("sensor_tick", str(sensor_dict["sensor_tick"]))
        sensor_bp.set_attribute("rotation_frequency", "0")
        sensor_bp.set_attribute("horizontal_fov", "0")
        for location in sensor_locations:
            # Place lidars at each sensor location
            transform = carla.Transform(
                carla.Location(
                    x=location[0][0],
                    y=location[0][1],
                    z=vehicle_height,
                ),
                carla.Rotation(
                    pitch=0.0,
                    yaw=location[1],
                    roll=0.0,
                ),
            )
            sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)
            sensors.append((sensor, location[1]))
        return sensors

    def callback(self, point_cloud, angle: float):
        global configuration
        detections = sorted(
            point_cloud,
            key=lambda detection: detection.point.x**2 + detection.point.y**2,
        )

        driveable_index = None
        obstacle_idx = 0
        for i in range(len(detections)):
            if detections[i].object_tag in self.__drivable_tags:
                driveable_index = i
            else:
                obstacle_idx = detections[i].object_idx
                break

        if driveable_index is None:
            driveable_distance = 0
        else:
            driveable_distance = math.sqrt(
                detections[driveable_index].point.x ** 2
                + detections[driveable_index].point.y ** 2
            )

        index = int(round(angle * len(self.__sensors) / 360))
        self.__circogram[index] = driveable_distance

        # Static obstacle
        if obstacle_idx == 0:
            velocity_x = 0
            velocity_y = 0
        else:
            velocity_x = self.__world.get_actor(obstacle_idx).get_velocity().x
            velocity_y = self.__world.get_actor(obstacle_idx).get_velocity().y
        self.__circogram_velocity_x[index] = velocity_x
        self.__circogram_velocity_y[index] = velocity_y

        self.__sensor_ready_list[index] = True

    def get_last_data(self):
        image = np.zeros((360, 640))
        angle_diff = 360 / len(self.__circogram)
        for i, distance in enumerate(self.__circogram):
            if distance is not None:
                angle = i * angle_diff
                x = int(distance * 5 * math.cos(math.radians(angle)))
                y = int(distance * 5 * math.sin(math.radians(angle)))
                cv2.line(image, (320, 180), (320 + y, 180 - x), (255, 255, 255), 1)

        return image

    def get_data(self):
        circogram_array = np.array(self.__circogram).reshape(-1, 1)
        velocity_x_array = np.array(self.__circogram_velocity_x).reshape(-1, 1)
        velocity_y_array = np.array(self.__circogram_velocity_y).reshape(-1, 1)
        combined_array = np.hstack(
            (circogram_array, velocity_x_array, velocity_y_array)
        )
        return combined_array

    def is_ready(self):
        return all(self.__sensor_ready_list)

    def destroy(self):
        for sensor, _ in self.__sensors:
            sensor.destroy()
