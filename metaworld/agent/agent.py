from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from metaworld.policies import ENV_POLICY_MAP
from metaworld.policies.policy import Policy
from metaworld.sawyer_xyz_env import SawyerXYZEnv


class MetaworldAgent(ABC):
    @abstractmethod
    def get_action(self, env: gym.Env, obs: npt.NDArray[np.float64], info: dict) -> npt.NDArray[np.float32]:
        """Get action for a given observation in a given environment.

        Args:
            env (gym.Env): The environment instance. No modifications should be made to it.
                           It is provided only for reference (e.g., to access the action space
                           or to get the environment name via `env.unwrapped.ENV_NAME`).
            obs (np.ndarray): The current observation from the environment.
            info (dict): Additional information from the environment.
            env_name (str): The name of the environment/task.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state of the agent between episodes."""
        pass


class RandomMetaworldAgent(MetaworldAgent):
    def __init__(self, seed: int | None = None) -> None:
        if seed is None:
            self.seed = 42
        self.seed = seed
        self.reset()

    def get_action(self, env: gym.Env, obs: npt.NDArray[np.float64], info: dict) -> npt.NDArray[np.float32]:
        action_space = cast(gym.spaces.Box, env.action_space)
        low = action_space.low
        high = action_space.high
        return self.rng.uniform(low, high).astype(np.float32)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


class ExpertPolicyMetaworldAgent(MetaworldAgent):
    def __init__(self) -> None:
        self.policy: Policy | None = None
        self.policy_task_name: str | None = None

    def get_action(self, env: gym.Env, obs: npt.NDArray[np.float64], info: dict) -> npt.NDArray[np.float32]:
        env_name = cast(SawyerXYZEnv, env.unwrapped).ENV_NAME
        if self.policy is None or self.policy_task_name != env_name:
            self.policy_task_name = env_name
            policy_cls = ENV_POLICY_MAP[env_name]
            self.policy = policy_cls()
        return self.policy.get_action(obs)

    def reset(self):
        pass
