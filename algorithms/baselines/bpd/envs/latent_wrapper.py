from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast
from typing_extensions import TypedDict
from gym import spaces
import numpy as np

from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import _global_registry, ENV_CREATOR, register_env


class LatentWrapperConfig(TypedDict):
    env: str
    """Name of the environment to wrap."""

    env_config: dict
    """Configuration to be passed to the wrapped environment."""

    latent_dist: Callable[[], np.ndarray]
    """Thunk to use to draw random variables."""

    episodes_per_latent: int
    """Number of episodes that should be played for each latent vector sampled."""

    agents_with_latent: Sequence[AgentID]
    """The set of agent IDs for which latent vectors should be appended."""

    random_policy_dist: Optional[Callable[[int], np.ndarray]]
    """If this is not None, then also append the action probabilities at the current
    state of a random policy. This function should take a seed and return a random
    draw which is deterministic given the seed."""

    use_tuple: bool
    """Add latents as the second element of an obervation tuple instead of concatenating
    it to the observation tensor."""


ObsWithLatent = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


class LatentEnvWrapper(MultiAgentEnv):
    """
    Environment wrapper that appends a latent vector to the observations for each
    episode. It will maintain the same latent vector for a certain number of episodes
    before switching to a new one.
    """

    base_env: MultiAgentEnv
    config: LatentWrapperConfig

    current_latent: np.ndarray
    current_latent_id: int
    episodes_with_current_latent: int

    def __init__(self, config: LatentWrapperConfig):
        base_env_name = config["env"]
        base_env_config = config["env_config"]
        env_creator = _global_registry.get(ENV_CREATOR, base_env_name)
        base_env = env_creator(base_env_config)
        if isinstance(base_env, MultiAgentEnv):
            self.base_env = base_env
        else:
            env_creator = make_multi_agent(env_creator)
            self.base_env = env_creator(base_env_config)

        self.config = config
        self.config.setdefault("random_policy_dist", None)
        self.config.setdefault("use_tuple", False)

        if hasattr(self.base_env, "observation_space"):
            total_latent_size = self.config["latent_dist"]().shape[0]
            if self.config["random_policy_dist"] is not None:
                total_latent_size += self.config["random_policy_dist"](0).shape[0]
            if self.config["use_tuple"]:
                base_obs_space: spaces.Space = cast(
                    Any, self.base_env
                ).observation_space
                latent_bound = np.inf * np.ones(total_latent_size)
                latent_space = spaces.Box(low=-latent_bound, high=latent_bound)
                self.observation_space = spaces.Tuple((base_obs_space, latent_space))
            else:
                base_obs_space_box: spaces.Box = cast(
                    Any, self.base_env
                ).observation_space
                latent_bound = np.ones(
                    base_obs_space_box.shape[:-1] + (total_latent_size,)
                )
                self.observation_space = spaces.Box(
                    low=np.concatenate(
                        [base_obs_space_box.low, latent_bound * -np.inf], axis=-1
                    ),
                    high=np.concatenate(
                        [base_obs_space_box.high, latent_bound * np.inf], axis=-1
                    ),
                )

        self.action_space = self.base_env.action_space

        self._sample_latent()

    def __getattr__(self, name: str) -> Any:
        # Pass through any attribute accesses to the base env.
        return getattr(self.base_env, name)

    def _sample_latent(self) -> None:
        self.current_latent = self.config["latent_dist"]()
        self.current_latent_id = hash(np.random.random())
        self.episodes_with_current_latent = 0

    def reset(self):
        if self.episodes_with_current_latent >= self.config["episodes_per_latent"]:
            self._sample_latent()
        self.episodes_with_current_latent += 1

        base_obs = self.base_env.reset()
        return self._append_latent_to_obs_dict(base_obs)

    def step(self, action_dict):
        base_obs, base_reward, base_done, base_infos = self.base_env.step(action_dict)

        return (
            self._append_latent_to_obs_dict(base_obs),
            base_reward,
            base_done,
            base_infos,
        )

    def _append_latent_to_obs_dict(
        self,
        obs_dict: Dict[AgentID, np.ndarray],
    ) -> Dict[AgentID, ObsWithLatent]:
        obs_dict_with_latent: Dict[AgentID, ObsWithLatent] = {}
        for agent_id, obs in obs_dict.items():
            if agent_id in self.config["agents_with_latent"]:
                obs_dict_with_latent[agent_id] = self._append_latent_to_obs(obs)
            else:
                obs_dict_with_latent[agent_id] = obs
        return obs_dict_with_latent

    def _append_latent_to_obs(self, obs: np.ndarray) -> ObsWithLatent:
        latent_to_append = self.current_latent

        if self.config["random_policy_dist"] is not None:
            obs_hash = hash(obs.data.tobytes())
            random_policy_action_probs = self.config["random_policy_dist"](
                obs_hash + self.current_latent_id
            )
            latent_to_append = np.concatenate(
                [latent_to_append, random_policy_action_probs]
            )

        if self.config["use_tuple"]:
            return (obs, latent_to_append)
        else:
            latent_to_append = latent_to_append[
                (np.newaxis,) * (obs.ndim - self.current_latent.ndim) + (slice(None),)
            ]
            latent_to_append = np.broadcast_to(
                latent_to_append,
                obs.shape[:-1] + (latent_to_append.shape[-1],),
            )
            return cast(np.ndarray, np.concatenate([obs, latent_to_append], axis=-1))


register_env("latent_wrapper", lambda config: LatentEnvWrapper(config))
