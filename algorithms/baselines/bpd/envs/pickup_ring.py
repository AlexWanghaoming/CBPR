from typing import Any, Dict, Tuple
from typing_extensions import TypedDict
from ray.tune.registry import register_env

import gym
from gym import spaces
import numpy as np


class PickupRingConfigDict(TypedDict):
    ring_size: int
    horizon: int


class PickupRingEnv(gym.Env):
    """
    Simple environment for demonstrating the Boltzmann policy distribution. The agent
    lives in a ring with two actions (left and right). Going in one direction long
    enough (the "ring size") brings one back to the start state. The agent receives
    reward for picking up objects from halfway around the ring and bringing them back
    to the start state.
    """

    current_position: int
    holding_object: bool
    timestep: int

    def __init__(self, config: PickupRingConfigDict):
        super().__init__()
        self.config = config

        self.obs_space_size = self.config["ring_size"] + 1
        self.observation_space = spaces.Box(
            low=np.zeros(self.obs_space_size), high=np.ones(self.obs_space_size)
        )
        self.action_space = spaces.Discrete(2)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_space_size)
        obs[self.current_position] = 1
        if self.holding_object:
            obs[-1] = 1
        return obs

    def reset(self) -> np.ndarray:
        self.current_position = 0
        self.holding_object = False
        self.timestep = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action == 0:
            self.current_position += 1
            if self.current_position >= self.config["ring_size"]:
                self.current_position -= self.config["ring_size"]
        elif action == 1:
            self.current_position -= 1
            if self.current_position < 0:
                self.current_position += self.config["ring_size"]
        else:
            assert False, "Invalid action"

        reward = 0
        if self.current_position == 0 and self.holding_object:
            # Drop off object.
            reward = 1
            self.holding_object = False
        elif (
            self.current_position == self.config["ring_size"] // 2
            and not self.holding_object
        ):
            # Pick up object.
            self.holding_object = True

        self.timestep += 1
        done = self.timestep >= self.config["horizon"]

        return self._get_obs(), float(reward), done, {}


register_env("pickup_ring", lambda env_config: PickupRingEnv(env_config))
