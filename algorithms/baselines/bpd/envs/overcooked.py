from bpd.training_utils import load_trainer
import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    Callable,
    cast,
)
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../src/')
from typing_extensions import Literal, TypedDict
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedState,
    PlayerState,
    Direction,
    OvercookedGridworld,
    ObjectState,
    SoupState,
)
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import ENV_CREATOR, register_env, _global_registry
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor

from gym import spaces
import copy
import numpy as np
import inspect
import random


class RlLibAgent(Agent):
    """
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """

    other_player_action: int
    prev_obs: Optional[np.ndarray]
    human_data_to_condition_on: Optional[List[SampleBatch]]
    current_human_data: Optional[SampleBatch]

    def __init__(
        self,
        policy: TorchPolicy,
        agent_index,
        featurize_fn,
        human_data_to_condition_on: Optional[List[SampleBatch]] = None,
    ):
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn
        self.prev_action = 0
        self.prev_obs = None
        self.other_player_action = 0

        self.human_data_to_condition_on = human_data_to_condition_on
        if self.human_data_to_condition_on:
            for human_data in self.human_data_to_condition_on:
                human_data["prev_obs"] = np.roll(human_data[SampleBatch.OBS], 1, 0)
                human_data["prev_obs"][0, :] = 0
        self.current_human_data = None

        obs_space = self.policy.observation_space
        while getattr(obs_space, "original_space", None) is not None:
            obs_space = obs_space.original_space
        self.preprocessor: Preprocessor = get_preprocessor(obs_space)(obs_space)

    def reset(self):
        # Get initial rnn states and add batch dimension to each
        assert self.policy.model is not None
        if hasattr(self.policy.model, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state.cpu(), axis=0)
                for state in self.policy.model.get_initial_state()
            ]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.get_initial_state()
            ]
        else:
            self.rnn_state = []

        if self.human_data_to_condition_on:
            # Run human data through the network, get the resulting state, and save
            # it so we use for calculating actions.
            human_data = random.choice(self.human_data_to_condition_on)

            if self.rnn_state and len(self.rnn_state[0]) > 1:
                import torch

                state_batches = [
                    torch.from_numpy(state).to(self.policy.device)
                    for state in self.rnn_state
                ]
                assert isinstance(self.policy.model, TorchModelV2)
                _, states = self.policy.model(
                    self.policy._lazy_tensor_dict(human_data),
                    state_batches,
                    seq_lens=np.array([len(human_data)]),
                )
                self.rnn_state = [state.detach().cpu().numpy() for state in states]
            else:
                self.current_human_data = human_data

    def set_other_player_action(self, action: Action):
        """
        Set the previous action taken by the other player; some policies need this.
        """

        self.other_player_action = Action.ACTION_TO_INDEX[action]

    def _add_dim_to_obs(
        self, obs: Union[Tuple[np.ndarray, ...], np.ndarray]
    ) -> Union[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(obs, tuple):
            return tuple(obs_part[np.newaxis] for obs_part in obs)
        else:
            return obs[np.newaxis]

    def _compute_action_helper(
        self, state: OvercookedState, store_prev_obs=False
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        prev_obs = self.prev_obs
        if prev_obs is None:
            prev_obs = np.zeros_like(my_obs)

        input_dict = SampleBatch(
            {
                SampleBatch.CUR_OBS: self.preprocessor.transform(my_obs)[np.newaxis],
                SampleBatch.PREV_ACTIONS: np.array([self.prev_action]),
                "prev_obs": self.preprocessor.transform(prev_obs)[np.newaxis],
                "infos": np.array([{"other_action": self.other_player_action}]),
            }
        )
        if self.current_human_data is not None:
            import torch

            input_dict = cast(
                SampleBatch,
                SampleBatch.concat_samples(
                    [
                        SampleBatch(
                            {
                                k: v
                                for k, v in self.current_human_data.items()
                                if k in input_dict
                            }
                        ),
                        input_dict,
                    ]
                ),
            )
            state_batches = [
                torch.from_numpy(state).to(self.policy.device)[None]
                for state in self.policy.get_initial_state()
            ]
            assert self.policy.model is not None
            action_dist_inputs, state_out = self.policy.model(
                self.policy._lazy_tensor_dict(input_dict),
                state_batches,
                seq_lens=np.array([len(input_dict)]),
            )
            action_dist_inputs = action_dist_inputs[-1:]
            state_out = [state[-1:] for state in state_out]
            extra_fetches = {SampleBatch.ACTION_DIST_INPUTS: action_dist_inputs}
            assert self.policy.dist_class is not None
            action_dist = self.policy.dist_class(action_dist_inputs, self.policy.model)
            actions = action_dist.sample()
            results: Tuple[
                TensorType, List[TensorType], Dict[str, TensorType]
            ] = convert_to_numpy((actions, state_out, extra_fetches))
        else:
            for state_index, state in enumerate(self.rnn_state):
                input_dict[f"state_in_{state_index}"] = state

            results = self.policy.compute_actions_from_input_dict(input_dict)

        if store_prev_obs:
            self.prev_obs = my_obs

        return results

    def action_probabilities(self, state: OvercookedState):
        """
        Arguments: the same as action(), see below
        returns:
            - Normalized action probabilities determined by self.policy
        """

        # Compute non-normalized log probabilities from the underlying model
        actions, state_out, extra_fetches = self._compute_action_helper(state)
        logits = extra_fetches["action_dist_inputs"]

        # Softmax in numpy to convert logits to normalized probabilities
        return softmax(logits)

    def action(self, state: OvercookedState):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of
              the environment
        returns:
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """

        # Use Rllib.Policy class to compute action argmax and action probabilities
        [action_idx], state_out, extra_fetches = self._compute_action_helper(
            state, store_prev_obs=True
        )
        agent_action = Action.INDEX_TO_ACTION[action_idx]

        self.prev_action = action_idx

        # Softmax in numpy to convert logits to normalized probabilities
        logits = extra_fetches["action_dist_inputs"]
        action_probabilities = softmax(logits)

        agent_action_info = {"action_probs": action_probabilities}
        self.rnn_state = state_out

        return agent_action, agent_action_info

    @classmethod
    def from_checkpoint(cls, checkpoint_path, run="PPO", agent_index=0, policy_id=None):
        trainer = load_trainer(checkpoint_path, run)
        if policy_id is None:
            if trainer.config["multiagent"]["policies_to_train"]:
                policy_id = trainer.config["multiagent"]["policies_to_train"][0]
            else:
                policy_id = next(iter(trainer.config["multiagent"]["policies"]))
        env = _global_registry.get(ENV_CREATOR, trainer.config["env"])(
            trainer.config["env_config"]
        )
        policy = trainer.get_policy(policy_id)
        assert isinstance(policy, TorchPolicy)
        return cls(
            policy=policy,
            agent_index=agent_index,
            featurize_fn=env.featurize_fn_map["bc" if run == "bc" else "ppo"],
        )


class AgentPairWithVisibleActions(AgentPair):
    """
    Simple extension of AgentPair that allows players to see each other's previous
    actions.
    """

    prev_actions: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.a0 != self.a1
        self.prev_actions = (Action.INDEX_TO_ACTION[0], Action.INDEX_TO_ACTION[0])

    def joint_action(self, state):
        if hasattr(self.a0, "set_other_player_action"):
            self.a0.set_other_player_action(self.prev_actions[1])
        if hasattr(self.a1, "set_other_player_action"):
            self.a1.set_other_player_action(self.prev_actions[0])
        joint_action_and_infos = super().joint_action(state)
        (action_0, _), (action_1, _) = joint_action_and_infos
        self.prev_actions = action_0, action_1
        return joint_action_and_infos


class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    action_space: spaces.Discrete
    # ppo_observation_space: spaces.Box
    bc_observation_space: spaces.Box

    # List of all agent types currently supported
    supported_agents = ["ppo", "bc"]

    # Default bc_schedule, includes no bc agent at any time
    bc_schedule = self_play_bc_schedule = [(0, 0), (float("inf"), 0)]

    # Default environment params used for creation
    DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params": {"layout_name": "cramped_room", "rew_shaping_params": {}},
        # To be passed into OvercookedEnv constructor
        "env_params": {"horizon": 400},
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": 0.0,
            "reward_shaping_horizon": 0,
            "bc_schedule": self_play_bc_schedule,
            "use_phi": True,
            "extra_rew_shaping": {
                "onion_dispense": 0,
                "dish_dispense": 0,
            },
        },
    }

    def __init__(
        self,
        base_env: OvercookedEnv,
        reward_shaping_factor=0.0,
        reward_shaping_horizon=0,
        bc_schedule=None,
        share_dense_reward=False,
        extra_rew_shaping=DEFAULT_CONFIG["multi_agent_params"]["extra_rew_shaping"],
        use_phi=True,
        regen_mdp=False,
        no_regular_reward=False,
        action_rewards: List[float] = [0] * Action.NUM_ACTIONS,
        **kwargs,
    ):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before
            adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor
            reaches zero through linear annealing
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the
            value of bc_factor at timestep t_i with linear interpolation in between the t_i
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        """

        super().__init__()

        if bc_schedule:
            self.bc_schedule = bc_schedule
        self._validate_schedule(self.bc_schedule)
        self.base_env = base_env
        # since we are not passing featurize_fn in as an argument, we create it here and check its validity
        self.featurize_fn_map = {
            "ppo": lambda state: self.base_env.featurize_state_mdp(state),
            "bc": lambda state: self.base_env.featurize_state_mdp(state),
        }
        self._validate_featurize_fns(self.featurize_fn_map)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self.share_dense_reward = share_dense_reward
        self.extra_rew_shaping = extra_rew_shaping
        self.regen_mdp = regen_mdp
        self.no_regular_reward = no_regular_reward
        self.action_rewards = action_rewards
        self._setup_observation_space()
        self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        self.anneal_bc_factor(0)
        self.reset(regen_mdp=True)

    def _validate_featurize_fns(self, mapping):
        assert "ppo" in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert (
                k in self.supported_agents
            ), "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert (
                len(get_required_arguments(v)) == 1
            ), "Featurize_fn value must accept exactly one argument"

    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]

        assert (
            len(schedule) >= 2
        ), "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all(
            [t >= 0 for t in timesteps]
        ), "All timesteps in schedule must be non-negative"
        assert all(
            [v >= 0 and v <= 1 for v in values]
        ), "All values in schedule must be between 0 and 1"
        assert (
            sorted(timesteps) == timesteps
        ), "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if schedule[-1][0] < float("inf"):
            schedule.append((float("inf"), schedule[-1][1]))

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        # ppo observation
        # featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(
        #     state
        # )
        # obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        # high = np.ones(obs_shape) * float("inf")
        # low = np.ones(obs_shape) * 0.0
        # self.ppo_observation_space = spaces.Box(low, high, dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * np.inf
        low = np.ones(obs_shape) * -np.inf
        self.bc_observation_space = spaces.Box(low, high, dtype=np.float32)

        self.observation_space = self.bc_observation_space

    def _get_featurize_fn(self, agent_id):

        return lambda state: self.base_env.featurize_state_mdp(state)
        raise ValueError("Unsupported agent type {0}".format(agent_id))

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]

        return ob_p0, ob_p1

    def _populate_agents(self):
        # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        agents = ["ppo"]

        # Coin flip to determine whether other agent should be ppo or bc
        other_agent = "bc" if np.random.uniform() < self.bc_factor else "ppo"
        agents.append(other_agent)

        # Ensure agent names are unique
        agents[0] = agents[0] + "_0"
        agents[1] = agents[1] + "_1"

        # Randomize starting indices
        np.random.shuffle(agents)

        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        prev_state: OvercookedState = copy.deepcopy(self.base_env.state)
        current_timestep = prev_state.timestep

        action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment

        next_state: OvercookedState
        dense_reward: List[float]
        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=True
            )
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = [potential, potential]
        else:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=False
            )
            dense_reward = info["shaped_r_by_agent"]

        ob_p0, ob_p1 = self._get_obs(next_state)

        # Add some extra reward shaping.
        for object_type in ["onion", "dish"]:
            for player_index, pickup_timesteps in enumerate(
                self.base_env.game_stats[f"{object_type}_pickup"]
            ):
                if (
                    len(pickup_timesteps) > 0
                    and pickup_timesteps[-1] == current_timestep
                ):
                    # Make sure player is not facing an empty counter.
                    player_state: PlayerState = prev_state.players[player_index]
                    facing_pos = Action.move_in_direction(
                        player_state.position, player_state.orientation
                    )
                    if self.base_env.mdp.get_terrain_type_at_pos(facing_pos) != "X":
                        dense_reward[player_index] += self.extra_rew_shaping[
                            f"{object_type}_dispense"
                        ]

        # Caculate some extra custom metrics.
        prev_player_state: PlayerState
        info["custom_metrics"] = {}
        # if self.base_env.game_stats[f"onion_drop"][1]: import pdb; pdb.set_trace()
        for player_index, prev_player_state in enumerate(prev_state.players):
            next_player_state: PlayerState = next_state.players[player_index]
            if prev_player_state.has_object() and not next_player_state.has_object():
                facing_pos = Action.move_in_direction(
                    prev_player_state.position, prev_player_state.orientation
                )
                if self.base_env.mdp.get_terrain_type_at_pos(facing_pos) == "X":
                    x, y = facing_pos
                    object_type = prev_player_state.held_object.name
                    info["custom_metrics"][
                        f"{object_type}_drop_{x}_{y}_agent_{player_index}"
                    ] = 1

        shaped_reward = [sparse_reward, sparse_reward]
        for agent_index in range(2):
            if self.share_dense_reward:
                agent_dense_reward = sum(dense_reward)
            else:
                agent_dense_reward = dense_reward[agent_index]
            shaped_reward[agent_index] += (
                self.reward_shaping_factor * agent_dense_reward
            )

        for agent_index in range(2):
            if self.no_regular_reward:
                shaped_reward[agent_index] = 0.0
            agent_action = Action.ACTION_TO_INDEX[joint_action[agent_index]]
            shaped_reward[agent_index] += self.action_rewards[agent_action]

        obs = {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
        rewards = {
            self.curr_agents[0]: shaped_reward[0],
            self.curr_agents[1]: shaped_reward[1],
        }
        dones = {self.curr_agents[0]: done, self.curr_agents[1]: done, "__all__": done}
        infos = {self.curr_agents[0]: info.copy(), self.curr_agents[1]: info.copy()}

        # Report last of other agent's action to each agent in the info dict.
        infos[self.curr_agents[0]]["other_action"] = action[1]
        infos[self.curr_agents[1]]["other_action"] = action[0]

        return obs, rewards, dones, infos

    def reset(self, regen_mdp=None):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """

        if regen_mdp is None:
            regen_mdp = self.regen_mdp

        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state)
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(
            self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon
        )
        self.set_reward_shaping_factor(new_factor)

    def anneal_bc_factor(self, timesteps):
        """
        Set the current bc factor such that we anneal linearly until self.bc_factor_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        p_0 = self.bc_schedule[0]
        p_1 = self.bc_schedule[1]
        i = 2
        while timesteps > p_1[0] and i < len(self.bc_schedule):
            p_0 = p_1
            p_1 = self.bc_schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timesteps, end_t, end_v, start_t)
        self.set_bc_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def set_bc_factor(self, factor):
        self.bc_factor = factor

    def seed(self, seed):
        """
        set global random seed to make environment deterministic
        """
        # Our environment is already deterministic
        pass

    def render(self, mode):
        pass

    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert (
            env_config
            and "env_params" in env_config
            and "multi_agent_params" in env_config
        )
        assert (
            "mdp_params" in env_config or "mdp_params_schedule_fn" in env_config
        ), "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(
            mdp_params, env_params, outer_shape, mdp_params_schedule_fn
        )
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)


register_env(
    "overcooked_multi_agent",
    lambda env_config: OvercookedMultiAgent.from_config(env_config),
)


class OvercookedCallbacks(DefaultCallbacks):
    custom_metrics_keys: Set[str] = set()

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies=None,
        episode,
        **kwargs,
    ) -> None:
        super().on_episode_step(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs,
        )

        env: OvercookedMultiAgent = base_env.get_unwrapped()[0]
        info = episode.last_info_for(env.curr_agents[0])
        if "custom_metrics" in info:
            for metric_name, metric_value in info["custom_metrics"].items():
                episode.custom_metrics.setdefault(metric_name, 0)
                episode.custom_metrics[metric_name] += metric_value

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data

        sparse_reward (int) - total reward from deliveries agent earned this episode
        shaped_reward (int) - total reward shaping reward the agent earned this episode
        """
        # Get rllib.OvercookedMultiAgentEnv refernce from rllib wraper
        env = base_env.get_unwrapped()[0]
        # Both agents share the same info so it doesn't matter whose we use, just use 0th agent's
        info_dict = episode.last_info_for(env.curr_agents[0])

        self.custom_metrics_keys |= episode.custom_metrics.keys()
        for key in self.custom_metrics_keys - episode.custom_metrics.keys():
            episode.custom_metrics[key] = 0

        ep_info = info_dict["episode"]
        game_stats = ep_info["ep_game_stats"]

        # List of episode stats we'd like to collect by agent
        stats_to_collect = EVENT_TYPES

        # Parse info dicts generated by OvercookedEnv
        tot_sparse_reward = ep_info["ep_sparse_r"]
        tot_shaped_reward = ep_info["ep_shaped_r"]

        # Store metrics where they will be visible to rllib for tensorboard logging
        episode.custom_metrics["sparse_reward"] = np.array(tot_sparse_reward)
        episode.custom_metrics["shaped_reward"] = np.array(tot_shaped_reward)

        # Store per-agent game stats to rllib info dicts
        for stat in stats_to_collect:
            stats = game_stats[stat]
            episode.custom_metrics[stat + "_agent_0"] = np.array(len(stats[0]))
            episode.custom_metrics[stat + "_agent_1"] = np.array(len(stats[1]))

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    # Executes at the end of a call to Trainer.train, we'll update environment params (like annealing shaped rewards)
    def on_train_result(self, trainer, result, **kwargs):
        # Anneal the reward shaping coefficient based on environment paremeters and current timestep
        timestep = result["timesteps_total"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_reward_shaping_factor(timestep)
            )
        )

        # Anneal the bc factor based on environment paremeters and current timestep
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.anneal_bc_factor(timestep))
        )

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        pass


def build_overcooked_eval_function(
    eval_params,
    eval_mdp_params,
    env_params,
    outer_shape,
    agent_0_policy_str="ppo",
    agent_1_policy_str="ppo",
    use_bc_featurize_fn=False,
):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    agent_1_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    Note: Agent policies are shuffled each time, so agent_0_policy_str and agent_1_policy_str are symmetric
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        print("Computing rollout of current trained policy")

        # Randomize starting indices
        policies = [agent_0_policy_str, agent_1_policy_str]
        np.random.shuffle(policies)
        agent_0_policy, agent_1_policy = policies

        # Get the corresponding rllib policy objects for each policy string name
        agent_0_policy = trainer.get_policy(agent_0_policy)
        agent_1_policy = trainer.get_policy(agent_1_policy)

        agent_0_feat_fn = agent_1_feat_fn = None
        if "bc" in policies or use_bc_featurize_fn:
            base_ae = get_base_ae(eval_mdp_params, env_params)
            base_env = base_ae.env
            bc_featurize_fn = lambda state: base_env.featurize_state_mdp(state)
            if policies[0] == "bc" or use_bc_featurize_fn:
                agent_0_feat_fn = bc_featurize_fn
            if policies[1] == "bc" or use_bc_featurize_fn:
                agent_1_feat_fn = bc_featurize_fn

        # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this
        # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        results = evaluate(
            eval_params,
            eval_mdp_params,
            outer_shape,
            agent_0_policy,
            agent_1_policy,
            agent_0_feat_fn,
            agent_1_feat_fn,
        )

        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics["average_sparse_reward"] = np.mean(results["ep_returns"])
        return metrics

    return _evaluate


class AgentInformation(TypedDict):
    action_probs: np.ndarray


class EpisodeDict(TypedDict):
    ep_game_stats: dict
    ep_length: int
    ep_shaped_r: int
    ep_shaped_r_by_agent: np.ndarray
    ep_sparse_r: int
    ep_sparse_r_by_agent: np.ndarray


class EpisodeInformation(TypedDict):
    agent_infos: List[AgentInformation]
    phi_s: None
    phi_s_prime: None
    shaped_r_by_agent: List[int]
    sparse_r_by_agent: List[int]
    episode: EpisodeDict


OvercookedAction = Union[Literal["interact"], Tuple[int, int]]


class EvaluationResults(TypedDict):
    env_params: Sequence[dict]
    ep_actions: Sequence[Sequence[Tuple[OvercookedAction, OvercookedAction]]]
    ep_dones: Sequence[Sequence[bool]]
    ep_infos: Sequence[Sequence[EpisodeInformation]]
    ep_lengths: Sequence[int]
    ep_returns: Sequence[int]
    ep_rewards: Sequence[Sequence[int]]
    ep_states: Sequence[Sequence[OvercookedState]]
    mdp_params: Sequence[dict]
    metadatas: dict


def evaluate(
    eval_params,
    mdp_params: dict,
    outer_shape,
    agent_0_policy,
    agent_1_policy,
    agent_0_featurize_fn=None,
    agent_1_featurize_fn=None,
    env_params: dict = {},
    agent_0_params: dict = {},
    agent_1_params: dict = {},
) -> EvaluationResults:
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    print("eval mdp params", mdp_params)
    evaluator = get_base_ae(
        mdp_params,
        {"horizon": eval_params["ep_length"], "num_mdp": 1, **env_params},
        outer_shape,
    )

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = (
        agent_0_featurize_fn
        if agent_0_featurize_fn
        else evaluator.env.lossless_state_encoding_mdp
    )
    agent_1_featurize_fn = (
        agent_1_featurize_fn
        if agent_1_featurize_fn
        else evaluator.env.lossless_state_encoding_mdp
    )

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgent(
        agent_0_policy,
        agent_index=0,
        featurize_fn=agent_0_featurize_fn,
        **agent_0_params,
    )
    agent1 = RlLibAgent(
        agent_1_policy,
        agent_index=1,
        featurize_fn=agent_1_featurize_fn,
        **agent_1_params,
    )

    # Compute rollouts
    if "store_dir" not in eval_params:
        eval_params["store_dir"] = None
    if "display_phi" not in eval_params:
        eval_params["display_phi"] = False
    results: EvaluationResults = evaluator.evaluate_agent_pair(
        AgentPairWithVisibleActions(agent0, agent1),
        num_games=eval_params["num_games"],
        display=eval_params["display"],
        dir=eval_params["store_dir"],
        display_phi=eval_params["display_phi"],
    )

    return results


def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T


def get_base_env(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.env


def get_base_mlam(
    mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None
):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.mlam


def get_base_ae(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    """
    mdp_params: one set of fixed mdp parameter used by the enviroment
    env_params: env parameters (horizon, etc)
    outer_shape: outer shape of the environment
    mdp_params_schedule_fn: the schedule for varying mdp params
    return: the base agent evaluator
    """
    assert (
        mdp_params is None or mdp_params_schedule_fn is None
    ), "either of the two has to be null"
    if type(mdp_params) == dict and "layout_name" in mdp_params:
        ae = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params, env_params=env_params
        )
    elif "num_mdp" in env_params:
        if np.isinf(env_params["num_mdp"]):
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_params,
                env_params=env_params,
                outer_shape=outer_shape,
                mdp_params_schedule_fn=mdp_params_schedule_fn,
            )
        else:
            ae = AgentEvaluator.from_mdp_params_finite(
                mdp_params=mdp_params,
                env_params=env_params,
                outer_shape=outer_shape,
                mdp_params_schedule_fn=mdp_params_schedule_fn,
            )
    else:
        # should not reach this case
        raise NotImplementedError()
    return ae


def get_littered_start_state_fn(
    num_littered_objects: int,
    mdp: OvercookedGridworld,
    base_start_state_fn: Optional[Callable[[], OvercookedState]] = None,
):
    def littered_start_state_fn() -> OvercookedState:
        if base_start_state_fn is None:
            state = mdp.get_standard_start_state()
        else:
            state = base_start_state_fn()
        reachable_counter_positions = []
        empty_positions = mdp.get_valid_player_positions()
        for counter_position in mdp.get_counter_locations():
            for direction in Direction.ALL_DIRECTIONS:
                adjacent_position = Action.move_in_direction(
                    counter_position, direction
                )
                if adjacent_position in empty_positions:
                    reachable_counter_positions.append(counter_position)

        for counter_position in random.sample(
            reachable_counter_positions, num_littered_objects
        ):
            littered_object = random.choice(
                [
                    ObjectState("onion", counter_position),
                    ObjectState("dish", counter_position),
                    SoupState.get_soup(counter_position, num_onions=1, finished=True),
                    SoupState.get_soup(counter_position, num_onions=2, finished=True),
                    SoupState.get_soup(counter_position, num_onions=3, finished=True),
                ]
            )
            state.add_object(littered_object, counter_position)
        return state

    return littered_start_state_fn


# Returns the required arguments as inspect.Parameter objects in a list
def get_required_arguments(fn):
    required = []
    params = inspect.signature(fn).parameters.values()
    for param in params:
        if (
            param.default == inspect.Parameter.empty
            and param.kind == param.POSITIONAL_OR_KEYWORD
        ):
            required.append(param)
    return required


def iterable_equal(a, b):
    if hasattr(a, "__iter__") != hasattr(b, "__iter__"):
        return False
    if not hasattr(a, "__iter__"):
        return a == b

    if len(a) != len(b):
        return False

    for elem_a, elem_b in zip(a, b):
        if not iterable_equal(elem_a, elem_b):
            return False

    return True


HumanData = Dict[str, List[List[Tuple[dict, List[OvercookedAction]]]]]


def load_human_trajectories_as_sample_batch(
    human_data_fname: str,
    layout_name: str,
    traj_indices: Optional[Set[int]] = None,
    featurize_fn_id: Literal["bc", "ppo"] = "bc",
    _log=None,
) -> SampleBatch:
    env = OvercookedMultiAgent.from_config(
        {
            "mdp_params": {
                "layout_name": layout_name,
            },
            "env_params": {"horizon": 1e10},
            "multi_agent_params": {},
        }
    )

    batch_builder = SampleBatchBuilder()
    with open(human_data_fname, "rb") as human_data_file:
        human_data: HumanData = pickle.load(human_data_file)
        layout_trajectories = human_data[layout_name]
        for traj_index, trajectory in enumerate(layout_trajectories):
            if traj_indices is not None and traj_index not in traj_indices:
                continue

            if _log is not None:
                _log.info(f"Converting trajectory {traj_index} to RLLib format")

            add_trajectory_to_builder(
                batch_builder,
                trajectory,
                agent_index=0,
                episode_id=traj_index,
                env=env,
                featurize_fn_id=featurize_fn_id,
            )
            add_trajectory_to_builder(
                batch_builder,
                trajectory,
                agent_index=1,
                episode_id=traj_index + 10**9,
                env=env,
                featurize_fn_id=featurize_fn_id,
            )

    return batch_builder.build_and_reset()


def add_trajectory_to_builder(
    batch_builder: SampleBatchBuilder,
    trajectory: List[Tuple[dict, list]],
    agent_index: int,
    episode_id: int,
    env: OvercookedEnv,
    featurize_fn_id: Literal["bc", "ppo"] = "bc",
    check_transitions: bool = False,
):
    base_env = env.base_env
    prev_action = 0
    for t, ((state_dict, actions), (new_state_dict, _)) in enumerate(
        zip(trajectory[:-1], trajectory[1:])
    ):
        state = OvercookedState.from_dict(state_dict)
        state.timestep = t
        new_state = OvercookedState.from_dict(new_state_dict)
        new_state.timestep = t + 1
        if check_transitions:
            base_env.state = state
            next_state, reward, done, info = base_env.step(actions)
            assert (
                new_state == next_state
            ), "States differed (expected vs actual): {}\n\nexpected dict: \t{}\nactual dict: \t{}\nactions: \t{}".format(
                base_env.display_states(new_state, next_state),
                new_state.to_dict(),
                next_state.to_dict(),
                actions,
            )

        action = Action.ACTION_TO_INDEX[actions[agent_index]]
        batch_builder.add_values(
            t=t,
            eps_id=episode_id,
            agent_index=agent_index,
            obs=env.featurize_fn_map[featurize_fn_id](state)[agent_index],
            actions=action,
            action_prob=1.0,
            action_logp=0.0,
            rewards=0.0,
            prev_actions=prev_action,
            prev_rewards=0.0,
            dones=t == len(trajectory) - 2,
            infos={},
            new_obs=env.featurize_fn_map[featurize_fn_id](new_state)[agent_index],
            advantages=0.0,
        )
        prev_action = action


OVERCOOKED_OBS_LAYERS: List[str] = [
    "player_1_loc",
    "player_0_loc",
    "player_1_orientation_0",
    "player_1_orientation_1",
    "player_1_orientation_2",
    "player_1_orientation_3",
    "player_0_orientation_0",
    "player_0_orientation_1",
    "player_0_orientation_2",
    "player_0_orientation_3",
    "pot_loc",
    "counter_loc",
    "onion_disp_loc",
    "tomato_disp_loc",
    "dish_disp_loc",
    "serve_loc",
    "onions_in_pot",
    "tomatoes_in_pot",
    "onions_in_soup",
    "tomatoes_in_soup",
    "soup_cook_time_remaining",
    "soup_done",
    "dishes",
    "onions",
    "tomatoes",
    "urgency",
]
