import os
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import torch
import glob

import ray
from ray.rllib.utils.typing import TrainerConfigDict, ModelConfigDict
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.agents.trainer import COMMON_CONFIG

from sacred import Experiment
from sacred import SETTINGS as sacred_settings

from bpd.envs.overcooked import (
    OvercookedMultiAgent,
    build_overcooked_eval_function,
    evaluate,
    load_human_trajectories_as_sample_batch,
)
from bpd.training_utils import build_logger_creator
from bpd.agents.bc import BCTrainer


ex = Experiment("train_overcooked_bc")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


@ex.config
def sacred_config():
    # Environment
    layout_name = "cramped_room"

    # Training
    num_workers = 0
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    sgd_minibatch_size = 64
    num_training_iters = 500  # noqa: F841
    lr = 1e-3
    use_bc_features = True

    # Model
    model_config: ModelConfigDict
    if use_bc_features:
        num_hidden_layers = 2
        size_hidden_layers = 64
        model_config = {
            "fcnet_hiddens": [size_hidden_layers] * num_hidden_layers,
            "fcnet_activation": "relu",
        }
    else:
        num_hidden_layers = 3
        size_hidden_layers = 64
        num_filters = 25
        num_conv_layers = 3
        model_config = {
            "custom_model": "overcooked_ppo_model",
            "vf_share_layers": True,
            "custom_model_config": {
                "num_hidden_layers": num_hidden_layers,
                "size_hidden_layers": size_hidden_layers,
                "num_filters": num_filters,
                "num_conv_layers": num_conv_layers,
            },
        }

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs"  # noqa: F841
    experiment_tag = None
    experiment_name_parts = ["bc", layout_name]
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841

    # Human data
    human_data_fname = None  # noqa: F841

    environment_params = {
        "mdp_params": {
            "layout_name": layout_name,
        },
        "env_params": {"horizon": 1},
        "multi_agent_params": {},
    }

    env = OvercookedMultiAgent.from_config(environment_params)

    # Validation
    validation_prop = 0

    # Evaluation
    evaluation_interval = 10
    evaluation_ep_length = 400
    evaluation_num_games = 10
    evaluation_display = False

    config: TrainerConfigDict = {  # noqa: F841
        "env": "overcooked_multi_agent",
        "env_config": environment_params,
        "num_workers": num_workers,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr,
        "lr_drop_patience": 5,
        "validation_prop": validation_prop,
        "num_gpus": num_gpus,
        "seed": seed,
        "framework": "torch",
        "evaluation_interval": evaluation_interval,
        "custom_eval_function": build_overcooked_eval_function(
            eval_params={
                "ep_length": evaluation_ep_length,
                "num_games": evaluation_num_games,
                "display": evaluation_display,
            },
            eval_mdp_params=environment_params["mdp_params"],
            env_params=environment_params["env_params"],
            outer_shape=None,
            agent_0_policy_str=DEFAULT_POLICY_ID,
            agent_1_policy_str=DEFAULT_POLICY_ID,
            use_bc_featurize_fn=use_bc_features,
        ),
        "multiagent": {
            "policies": {
                DEFAULT_POLICY_ID: (
                    None,
                    env.bc_observation_space
                    if use_bc_features
                    else env.ppo_observation_space,
                    env.action_space,
                    {
                        "model": model_config,
                    },
                )
            },
        },
    }

    if "disable_env_checking" in COMMON_CONFIG:
        config["disable_env_checking"] = True

    del env


def get_human_offline_data(
    human_data_fname: str, layout_name: str, *, use_bc_features=True, _log=None
) -> str:
    offline_data_fname = ".".join(human_data_fname.split(".")[:-1]) + "_" + layout_name
    if not use_bc_features:
        offline_data_fname += "_ppo_features"
    if len(glob.glob(os.path.join(offline_data_fname, "*.json"))) == 0:
        human_data_sample_batch = load_human_trajectories_as_sample_batch(
            human_data_fname,
            layout_name,
            featurize_fn_id="bc" if use_bc_features else "ppo",
            _log=_log,
        )
        if _log is not None:
            _log.info("Saving trajectories")
        offline_writer = JsonWriter(offline_data_fname)
        offline_writer.write(human_data_sample_batch)
    return offline_data_fname


@ex.automain
def main(
    config,
    log_dir,
    experiment_name,
    sgd_minibatch_size,
    num_training_iters,
    save_freq,
    human_data_fname,
    layout_name,
    use_bc_features,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config["input"] = get_human_offline_data(
        human_data_fname, layout_name, use_bc_features=use_bc_features, _log=_log
    )
    trainer: Trainer = BCTrainer(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    result = None
    for train_iter in range(num_training_iters):
        _log.info(f"Starting training iteration {train_iter}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            _log.info(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    _log.info(f"Saved final checkpoint to {checkpoint}")

    # Run an evaluation at the end of training.
    bc_policy = trainer.get_policy()
    env = OvercookedMultiAgent.from_config(
        {
            "mdp_params": {
                "layout_name": layout_name,
            },
            "env_params": {"horizon": 1e10},
            "multi_agent_params": {},
        }
    ).base_env
    if use_bc_features:
        featurize_fn = lambda state: env.featurize_state_mdp(state)
    else:
        featurize_fn = lambda state: env.lossless_state_encoding_mdp(state)
    evaluate(
        eval_params={
            "ep_length": 400,
            "num_games": 10,
            "display": False,
        },
        mdp_params={
            "layout_name": layout_name,
            # "rew_shaping_params": {},
        },
        outer_shape=None,
        agent_0_policy=bc_policy,
        agent_1_policy=bc_policy,
        agent_0_featurize_fn=featurize_fn,
        agent_1_featurize_fn=featurize_fn,
    )

    return result
