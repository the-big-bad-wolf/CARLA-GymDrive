from gymnasium import spaces
import numpy as np


# Change this according to your needs.
observation_shapes = {
    "circogram": (60, 3),
    "position": (2,),
    "target_position": (2,),
    "heading": (2,),
    "velocity": (2,),
    "speed": (1,),
    "current_waypoint_relative_position": (2,),
    "next_waypoint_relative_position": (2,),
    "previous_steering": (1,),
    "previous_throttle_brake": (1,),
}

max_speed = 30.0  # m/s
circogram_range = 50.0  # meters
circogram_low_row = np.array([0.0, -max_speed, -max_speed])
circogram_high_row = np.array([circogram_range, max_speed, max_speed])

situations_map = {"Road": 0, "Roundabout": 1, "Junction": 2, "Tunnel": 3}

observation_space = spaces.Dict(
    {
        "circogram": spaces.Box(
            low=np.stack([circogram_low_row for _ in range(60)]),
            high=np.stack([circogram_high_row for _ in range(60)]),
            dtype=np.float32,
        ),
        "velocity": spaces.Box(
            low=np.array([-max_speed, -max_speed]),
            high=np.array([max_speed, max_speed]),
            dtype=np.float32,
        ),
        "heading": spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        ),
        "current_waypoint_relative_position": spaces.Box(
            low=np.array([-300, -300]),
            high=np.array([300, 300]),
            dtype=np.float32,
        ),
        "next_waypoint_relative_position": spaces.Box(
            low=np.array([-300, -300]),
            high=np.array([300, 300]),
            dtype=np.float32,
        ),
        "target_relative_position": spaces.Box(
            low=np.array([-300, -300]),
            high=np.array([300, 300]),
            dtype=np.float32,
        ),
        "previous_steering": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32,
        ),
        "previous_throttle_brake": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32,
        ),
    }
)

# For continuous actions (steering [-1.0, 1.0], throttle/brake [-1.0, 1.0])
continuous_action_space = spaces.Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
)

# For discrete actions (accelerate, deaccelerate, turn left, turn right)
discrete_action_space = spaces.Discrete(4)
