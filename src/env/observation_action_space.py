from gymnasium import spaces
import numpy as np


# Change this according to your needs.
observation_shapes = {
    "circogram_distance": (60,),
    "circogram_velocity": (60, 2),
    "position": (2,),
    "target_position": (2,),
    "next_waypoint_relative_position": (2,),
    "speed": (1,),
    "heading": (1,),
    "previous_steering": (1,),
    "previous_throttle_brake": (1,),
}

situations_map = {"Road": 0, "Roundabout": 1, "Junction": 2, "Tunnel": 3}

observation_space = spaces.Dict(
    {
        "circogram_distance": spaces.Box(
            low=0,
            high=np.inf,  # UPDATE WHEN CHANGING CIRCOGRAM RANGE
            shape=observation_shapes["circogram_distance"],
            dtype=np.float32,
        ),
        "circogram_velocity": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["circogram_velocity"],
            dtype=np.float32,
        ),
        "next_waypoint_relative_position": spaces.Box(
            low=-np.inf,
            high=np.inf,  # May want to split this
            shape=observation_shapes["next_waypoint_relative_position"],
            dtype=np.float32,
        ),
        "speed": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["speed"],
            dtype=np.float32,
        ),
        "heading": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shapes["heading"],
            dtype=np.float32,
        ),
        "previous_steering": spaces.Box(
            low=-1,
            high=1,
            shape=observation_shapes["previous_steering"],
            dtype=np.float32,
        ),
        "previous_throttle_brake": spaces.Box(
            low=-1,
            high=1,
            shape=observation_shapes["previous_throttle_brake"],
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
