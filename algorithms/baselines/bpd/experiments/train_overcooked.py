from ray.rllib.policy.torch_policy import TorchPolicy
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from bpd.experiments.train_overcooked_bc import get_human_offline_data
from bpd.models.overcooked_models import OvercookedRecurrentStateModel
from typing import Any, Callable, List, Optional, Type, cast
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.agents.trainer import COMMON_CONFIG
from typing_extensions import Literal
from overcooked_ai_py.mdp.actions import Action
from logging import Logger
import torch
import numpy as np
from gym import spaces

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.typing import (
    MultiAgentPolicyConfigDict,
    PolicyID,
    TrainerConfigDict,
)
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.policy import PolicySpec

from sacred import Experiment
from sacred import SETTINGS as sacred_settings

from bpd.envs.overcooked import (
    OvercookedCallbacks,
    OvercookedMultiAgent,
    build_overcooked_eval_function,
)
from bpd.training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
    load_trainer_config,
)
from bpd.agents.bpd_trainer import BPDTrainer
from bpd.agents.distillation_prediction import (
    DistillationPredictionTrainer,
    DEFAULT_CONFIG as distillation_default_config,
)
from bpd.agents.gail import GailTrainer

ex = Experiment("train_overcooked")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


def make_overcooked_sacred_config(ex: Experiment):  # noqa
    @ex.config
    def sacred_config(_log):  # noqa
        run: Literal["ppo", "bpd", "distill", "gail"] = "ppo"

        # Environment
        layout_name = "cramped_room"
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }
        dispense_reward = 0
        horizon = 400
        no_regular_reward = False
        action_rewards = [0] * Action.NUM_ACTIONS

        # Training
        num_workers = 2
        seed = 0
        num_gpus = 1 if torch.cuda.is_available() else 0
        num_gpus_per_worker = 0
        train_batch_size = 12000
        sgd_minibatch_size = 2000
        rollout_fragment_length = horizon
        num_training_iters = 500  # noqa: F841
        lr = 1e-3
        grad_clip = 0.1
        gamma = 0.99
        gae_lambda = 0.98
        vf_share_layers = False
        vf_loss_coeff = 1e-4
        entropy_coeff_start = 0.2
        entropy_coeff_end = 1e-3
        entropy_coeff_horizon = 3e6
        kl_coeff = 0.2
        clip_param = 0.05
        num_sgd_iter = 8

        # Model
        num_hidden_layers = 3
        size_hidden_layers = 64
        num_filters = 25
        num_conv_layers = 3
        split_backbone = False
        use_lstm = False
        use_attention = False
        use_sequence_model = False
        lstm_cell_size = 256  # LSTM memory cell size (only used if use_lstm=True)
        latent_size = 10  # For use with OvercookedPPODistributionModel
        use_tuple_for_latent = True
        sequence_memory_inference = 1 if use_lstm else horizon
        ignore_latents = False
        max_seq_len = horizon
        custom_model_config = {
            "num_hidden_layers": num_hidden_layers,
            "size_hidden_layers": size_hidden_layers,
            "num_filters": num_filters,
            "num_conv_layers": num_conv_layers,
            "split_backbone": split_backbone,
        }
        custom_model = "overcooked_ppo_model"
        # Use the internal state of the recurrent model at this checkpoint as extra
        # input to the policy.
        recurrent_checkpoint = None
        recurrent_policy_id = "ppo_distilled"
        assert not (recurrent_checkpoint and use_sequence_model)
        if use_sequence_model:
            if run == "distill":
                custom_model = "overcooked_sequence_prediction_model"
            else:
                custom_model = "overcooked_sequence_policy_model"
                custom_model_config["memory_inference"] = sequence_memory_inference

        elif recurrent_checkpoint is not None:
            recurrent_trainer_config = load_trainer_config(recurrent_checkpoint)
            recurrent_policy_config = recurrent_trainer_config["multiagent"][
                "policies"
            ][recurrent_policy_id][3]
            custom_model = "overcooked_recurrent_state_model"
            custom_model_config["recurrent_model_config"] = recurrent_policy_config[
                "model"
            ]

        model_config = {
            "custom_model": custom_model,
            "max_seq_len": max_seq_len,
            "custom_model_config": custom_model_config,
            "vf_share_layers": vf_share_layers,
            "use_lstm": use_lstm,
            "lstm_cell_size": lstm_cell_size,
            "use_attention": use_attention,
        }
        if use_sequence_model:
            custom_model_config["use_lstm"] = use_lstm
            del model_config["use_lstm"]

        # Reward shaping
        use_phi = (
            False  # Whether dense reward should come from potential function or not
        )
        # Constant by which shaped rewards are multiplied by when calculating total reward
        reward_shaping_factor = 1.0
        # Linearly anneal the reward shaping factor such that it reaches zero after this
        # number of timesteps
        reward_shaping_horizon = 2.5e6
        # Whether the agents should both get all dense rewards.
        share_dense_reward = False

        # MaxEnt policy RL
        distill_random_policies = False
        temperature = 1.0
        prior_concentration = 1.0
        input_random_policy = distill_random_policies
        random_policy_dist: Optional[Callable[[int], np.ndarray]] = None
        if input_random_policy:
            def random_policy_dist(seed: int, concentration=prior_concentration):
                return np.log(
                    np.random.default_rng(seed=abs(seed)).dirichlet(
                        np.ones(Action.NUM_ACTIONS) * concentration
                    )
                )

        discriminate_sequences = False
        discriminator_input_smoothing_std = 0
        discriminator_scale = 1
        use_latent_attention = False
        latent_dist = lambda latent_size=latent_size: np.random.normal(
            0, 1, latent_size
        )
        checkpoint_to_load_policies = None
        loaded_policy_id = None
        # Set these parameters to use a human trajectory instead of sampling one:
        human_data_fname = None
        trajectory_index = 0
        latents_per_iteration = 4
        episodes_per_latent = (train_batch_size // latents_per_iteration) // horizon
        if run == "bpd" and (
            episodes_per_latent * latents_per_iteration * horizon != train_batch_size
            or num_workers * (latents_per_iteration // max(num_workers, 1))
            != latents_per_iteration
        ):
            _log.warning("Uneven split of episodes or workers across latent vectors!")
        if run == "bpd":
            custom_model_config.update(
                {
                    "latent_size": latent_size,
                    "ignore_latents": ignore_latents,
                    "discriminate_sequences": discriminate_sequences,
                    "discriminator_scale": discriminator_scale,
                    "discriminator_input_smoothing_std": discriminator_input_smoothing_std,
                    "pointless_discriminator_latent_input": False,
                    "use_latent_attention": use_latent_attention,
                }
            )
            model_config["custom_model"] = "overcooked_ppo_distribution_model"


        if checkpoint_to_load_policies is not None:
            checkpoint_to_load_policies_config: TrainerConfigDict = load_trainer_config(
                checkpoint_to_load_policies
            )

        # GAIL
        demonstration_num_episodes = 4

        # Multiagent
        multiagent_mode: Literal["self_play", "cross_play"] = "self_play"
        policy_ids: List[str]
        policy_mapping_fn: Callable[[str], str]
        if multiagent_mode == "self_play":
            policy_ids = ["ppo"]
            policy_mapping_fn = lambda agent_id: "ppo"
        elif multiagent_mode == "cross_play":
            policy_ids = ["ppo_0", "ppo_1"]
            policy_mapping_fn = lambda agent_id: agent_id
        policies_to_train = policy_ids

        # Evaluation
        evaluation_interval = 10**20 if run in ["bpd", "distill"] else 50
        evaluation_ep_length = horizon
        evaluation_num_games = 1
        evaluation_display = False

        # Logging
        save_freq = 25  # noqa: F841
        log_dir = "data/logs"  # noqa: F841
        experiment_tag = None
        experiment_name_parts = [multiagent_mode, run, layout_name]
        if no_regular_reward:
            experiment_name_parts.append("no_regular_reward")
        if any(reward != 0 for reward in action_rewards):
            experiment_name_parts.append(
                "action_rewards_"
                + "_".join([f"{reward:.2g}" for reward in action_rewards])
            )
        if run == "bpd":
            experiment_name_parts.append(f"temperature_{temperature}")
            experiment_name_parts.append(f"concentration_{prior_concentration}")
        if human_data_fname is not None:
            experiment_name_parts.append(f"human_{trajectory_index}")
        if experiment_tag is not None:
            experiment_name_parts.append(experiment_tag)
        experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
        checkpoint_path = None  # noqa: F841

        input = "sampler"

        env_id = "overcooked_multi_agent"
        env_config = {
            # To be passed into OvercookedGridWorld constructor
            "mdp_params": {
                "layout_name": layout_name,
                "rew_shaping_params": rew_shaping_params,
            },
            # To be passed into OvercookedEnv constructor
            "env_params": {"horizon": horizon},
            # To be passed into OvercookedMultiAgent constructor
            "multi_agent_params": {
                "reward_shaping_factor": reward_shaping_factor,
                "reward_shaping_horizon": reward_shaping_horizon,
                "use_phi": use_phi,
                "share_dense_reward": share_dense_reward,
                "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
                "extra_rew_shaping": {
                    "onion_dispense": dispense_reward,
                    "dish_dispense": dispense_reward,
                },
                "no_regular_reward": no_regular_reward,
                "action_rewards": action_rewards,
            },
        }

        overcooked_env_config = env_config
        env = OvercookedMultiAgent.from_config(overcooked_env_config)

        policies: MultiAgentPolicyConfigDict = {}

        ppo_observation_space = env.ppo_observation_space
        if run == "bpd" or distill_random_policies:
            # Add latent vector to observation space.
            total_latent_size = latent_size
            if input_random_policy:
                total_latent_size += Action.NUM_ACTIONS
            if use_tuple_for_latent:
                latent_bound = np.inf * np.ones(total_latent_size)
                latent_space = spaces.Box(low=-latent_bound, high=latent_bound)
                ppo_observation_space = spaces.Tuple(
                    (ppo_observation_space, latent_space)
                )
            else:
                assert isinstance(ppo_observation_space, spaces.Box)
                latent_bound = np.ones_like(ppo_observation_space.low[..., :1])
                latent_bound = latent_bound.repeat(total_latent_size, axis=-1)
                ppo_observation_space = spaces.Box(
                    low=np.concatenate(
                        [ppo_observation_space.low, latent_bound * -np.inf], axis=-1
                    ),
                    high=np.concatenate(
                        [ppo_observation_space.high, latent_bound * np.inf], axis=-1
                    ),
                )

        for policy_id in policy_ids:
            policies[policy_id] = PolicySpec(
                None,
                ppo_observation_space,
                env.action_space,
                {"model": model_config},
            )


        if multiagent_mode == "cross_play" and checkpoint_to_load_policies is not None:
            # In the case where we want to train a policy via cross-play with an
            # existing policy from a checkpoint. The loaded policy does not
            # get trained.

            bc_features = False
            if "multiagent" in checkpoint_to_load_policies_config:
                loaded_policy_dict: MultiAgentPolicyConfigDict = (
                    checkpoint_to_load_policies_config["multiagent"]["policies"]
                )
                if loaded_policy_id is None:
                    loaded_policy_ids = list(loaded_policy_dict.keys())
                    assert len(loaded_policy_ids) == 1
                    (loaded_policy_id,) = loaded_policy_ids
                loaded_policy_obs_space: spaces.Box = loaded_policy_dict[
                    loaded_policy_id
                ][1]
                bc_features = (
                    loaded_policy_obs_space.shape == env.bc_observation_space.shape
                )
            else:
                bc_features = True
                loaded_policy_id = DEFAULT_POLICY_ID
                loaded_policy_dict = {}

            if not bc_features:
                (
                    loaded_policy_cls,
                    loaded_policy_obs_space,
                    loaded_policy_action_space,
                    loaded_policy_config,
                ) = loaded_policy_dict[loaded_policy_id]
                policies[loaded_policy_id] = PolicySpec(
                    None,
                    loaded_policy_obs_space,
                    loaded_policy_action_space,
                    loaded_policy_config,
                )
                policy_mapping_fn = (
                    lambda agent_id, loaded_policy_id=loaded_policy_id: "ppo_0"
                    if agent_id == "ppo_0"
                    else loaded_policy_id
                )

                checkpoint_env_config = checkpoint_to_load_policies_config["env_config"]
                if checkpoint_to_load_policies_config["env"] == "latent_wrapper":
                    env_config = {
                        "env": env_id,
                        "env_config": env_config,
                        "latent_dist": checkpoint_env_config["latent_dist"],
                        "episodes_per_latent": 1,
                        "agents_with_latent": {"ppo_1"},
                        "random_policy_dist": checkpoint_env_config[
                            "random_policy_dist"
                        ],
                        "use_tuple": checkpoint_env_config["use_tuple"],
                    }
                    env_id = "latent_wrapper"
            else:
                # We're doing cross play with a BC agent.
                assert loaded_policy_id == DEFAULT_POLICY_ID
                policies[DEFAULT_POLICY_ID] = (
                    None,
                    env.bc_observation_space,
                    env.action_space,
                    {
                        "model": {
                            **checkpoint_to_load_policies_config.get("model", {}),
                            **loaded_policy_dict.get(
                                DEFAULT_POLICY_ID, (None, None, None, {})
                            )[3].get("model", {}),
                            "vf_share_layers": True,
                        }
                    },
                )
                env_config["multi_agent_params"]["bc_schedule"] = [
                    (0, 1),
                    (float("inf"), 1),
                ]
                policy_mapping_fn = (
                    lambda agent_id: DEFAULT_POLICY_ID
                    if agent_id.startswith("bc")
                    else "ppo_0"
                )

        config: TrainerConfigDict = {  # noqa: F841
            "env": env_id,
            "env_config": env_config,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": policies_to_train,
            },
            "callbacks": OvercookedCallbacks,
            "custom_eval_function": build_overcooked_eval_function(
                eval_params={
                    "ep_length": evaluation_ep_length,
                    "num_games": evaluation_num_games,
                    "display": evaluation_display,
                },
                eval_mdp_params=overcooked_env_config["mdp_params"],
                env_params=overcooked_env_config["env_params"],
                outer_shape=None,
                agent_0_policy_str=policy_ids[0],
                agent_1_policy_str=policy_ids[-1],
            ),
            "num_workers": num_workers,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "num_sgd_iter": num_sgd_iter,
            "lr": lr,
            "grad_clip": grad_clip,
            "gamma": gamma,
            "lambda": gae_lambda,
            "vf_loss_coeff": vf_loss_coeff,
            "kl_coeff": kl_coeff,
            "clip_param": clip_param,
            "num_gpus": num_gpus,
            "num_gpus_per_worker": num_gpus_per_worker,
            "seed": seed,
            "evaluation_interval": evaluation_interval,
            "entropy_coeff_schedule": [
                (0, entropy_coeff_start),
                (entropy_coeff_horizon, entropy_coeff_end),
            ],
            "framework": "torch",
            "input": input,
            "input_evaluation": [],
        }

        if "disable_env_checking" in COMMON_CONFIG:
            config["disable_env_checking"] = True

        TrainerClass: Type[Trainer]
        if run == "bpd":
            TrainerClass = BPDTrainer  # noqa: F841
            config.update(
                {
                    "temperature": temperature,
                    "prior_concentration": prior_concentration,
                    "latent_size": latent_size,
                    "env": "latent_wrapper",
                    "env_config": {
                        "env": config["env"],
                        "env_config": config["env_config"],
                        "latent_dist": latent_dist,
                        "episodes_per_latent": episodes_per_latent,
                        "agents_with_latent": {"ppo_0", "ppo_1"},
                        "use_tuple": use_tuple_for_latent,
                    },
                }
            )
        del env


make_overcooked_sacred_config(ex)


@ex.automain
def main(
    config,
    log_dir,
    experiment_name,
    num_training_iters,
    save_freq,
    checkpoint_to_load_policies: Optional[str],
    policy_ids,
    TrainerClass: Type[Trainer],
    checkpoint_path: Optional[str],
    recurrent_checkpoint: Optional[str],
    recurrent_policy_id: str,
    _log: Logger,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer = TrainerClass(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    if checkpoint_to_load_policies is not None:
        _log.info(f"Initializing policies from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(checkpoint_to_load_policies, trainer)

    if checkpoint_path is not None:
        _log.info(f"Restoring checkpoint at {checkpoint_path}")
        trainer.restore(checkpoint_path)

    if recurrent_checkpoint is not None:
        _log.info(f"Loading recurrent model from {recurrent_checkpoint}")

        def load_recurrent_model_checkpoint(policy: Policy, policy_id: PolicyID):
            model = cast(TorchPolicy, policy).model
            if isinstance(model, OvercookedRecurrentStateModel):
                assert recurrent_checkpoint is not None
                model.load_recurrent_model_from_checkpoint(
                    recurrent_checkpoint, recurrent_policy_id
                )

        workers: WorkerSet = cast(Any, trainer).workers
        workers.foreach_policy(load_recurrent_model_checkpoint)

    result = None
    for _ in range(num_training_iters):
        _log.info(f"Starting training iteration {trainer.iteration}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            _log.info(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    _log.info(f"Saved final checkpoint to {checkpoint}")

    return result
