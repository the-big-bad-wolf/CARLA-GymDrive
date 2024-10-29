from gymnasium import spaces
import numpy as np

# Change this according to your needs.
observation_shapes = {
    "rgb_data": (360, 640, 3),
    "lidar_data": (3, 500),
    "circogram_data": (80, 3),
    "position": (3,),
    "target_position": (3,),
    "next_waypoint_position": (3,),
    "speed": (2,),
    "acceleration": (2,),
    "num_of_stuations": 4,
}

situations_map = {"Road": 0, "Roundabout": 1, "Junction": 2, "Tunnel": 3}

observation_space = spaces.Dict(
    {
        "circogram": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["circogram_data"],
            dtype=np.float32,
        ),
        "position": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["position"],
            dtype=np.float32,
        ),
        "target_position": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["target_position"],
            dtype=np.float32,
        ),
        "next_waypoint_position": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["next_waypoint_position"],
            dtype=np.float32,
        ),
        "speed": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "acceleration": spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        ),
    }
)

# For continuous actions (steering [-1.0, 1.0], throttle/brake [-1.0, 1.0])
continuous_action_space = spaces.Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
)

# For discrete actions (accelerate, deaccelerate, turn left, turn right)
discrete_action_space = spaces.Discrete(4)
