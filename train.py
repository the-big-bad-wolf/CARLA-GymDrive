"""
This is an example script to train a DQN agent in the Carla environment using the stable-baselines3 library.

This is just a simple example to show that it is possible to train an agent using the stable-baselines3 library.

If you want to train an agent for a more real-life problem, you should consider using more complex models and hyperparameters; or even using other RL libraries compatible with the gym interface.
"""

from src.env.environment import (
    CarlaEnv,
)  # It is mandatory to import the environment even if it is not used in this script
from stable_baselines3 import DQN, SAC, PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env():
    env = gym.make(
        "carla-rl-gym-v0",
        time_limit=45,
        initialize_server=False,
        random_weather=False,
        synchronous_mode=True,
        continuous=True,
        show_sensor_data=True,
        has_traffic=True,
        verbose=True,
    )
    return env


def main():
    # Create the environment
    env = DummyVecEnv([make_env])
    # Create the agent
    """     
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log="data/tensorboard/",
    )
     """
    model = PPO.load("models/candidate18_25000.zip", env=env)

    # Save the agent at regular intervals
    for i in range(1, 21):
        model.learn(
            total_timesteps=25000
        )  # This causes some issues the spawning actors when not restarting the CARLA server
        model.save(f"output/ppo_agent_{i*25000}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
