# All imports except rllib
import json
from typing import Any, Dict, Type, cast
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
import torch
import os

# Sacred setup (must be before rllib imports)
from sacred import Experiment

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from bpd.agents.bpd_policy import BPDPolicy
from bpd.online_bc import OnlineBCActionPredictor

from bpd.training_utils import load_trainer
from bpd.envs.overcooked import (
    OvercookedMultiAgent,
    load_human_trajectories_as_sample_batch,
)
from bpd.experiments.evaluate_overcooked import (
    RunStr,
    default_policy_ids,
    trainer_classes,
)
from bpd.latent_prediction import VIActionPredictor

ex = Experiment("evaluate_overcooked_prediction")


@ex.config
def config():
    run: RunStr = "bc"
    checkpoint_path = None  # noqa: F841
    policy_id = default_policy_ids[run]  # noqa: F841
    trajectory_index = None  # noqa: F841
    human_data_fname = None  # noqa: F841

    run_bc_online = False  # noqa: F841

    out_tag = None
    if out_tag is not None:
        out_path = os.path.join(os.path.dirname(checkpoint_path), out_tag)
    else:
        out_path = None  # noqa: F841


@ex.automain
def main(
    checkpoint_path,
    policy_id,
    run,
    human_data_fname,
    trajectory_index,
    run_bc_online: bool,
    out_path,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer: Trainer = load_trainer(
        checkpoint_path,
        run=trainer_classes[run],
        config_overrides={"input": "sampler"},
    )
    policy = trainer.get_policy(policy_id)
    assert isinstance(policy, TorchPolicy)
    assert policy.model is not None

    overcooked_env_config = trainer.config["env_config"]
    while "mdp_params" not in overcooked_env_config:
        overcooked_env_config = overcooked_env_config["env_config"]
    env = OvercookedMultiAgent.from_config(overcooked_env_config)
    human_data_sample_batch = load_human_trajectories_as_sample_batch(
        human_data_fname,
        layout_name=overcooked_env_config["mdp_params"]["layout_name"],
        traj_indices=None if trajectory_index is None else {trajectory_index},
        featurize_fn_id="bc"
        if policy.observation_space == env.bc_observation_space
        else "ppo",
        _log=_log,
    )

    stay = Action.ACTION_TO_INDEX[Action.STAY]

    episode_batches = human_data_sample_batch.split_by_episode()
    episode_cross_entropies: Dict[int, float] = {}
    episode_cross_entropies_no_stay: Dict[int, float] = {}
    episode_accuracies: Dict[int, float] = {}
    episode_accuracies_no_stay: Dict[int, float] = {}
    for episode_batch in episode_batches:
        episode_id = int(episode_batch[SampleBatch.EPS_ID][0])
        input_dict = policy._lazy_tensor_dict(episode_batch)
        input_dict["prev_obs"] = input_dict["obs"].roll(1, 0)
        input_dict["prev_obs"][0, :] = 0
        state_batches = [
            torch.from_numpy(state).to(policy.device)[None]
            for state in policy.get_initial_state()
        ]
        seq_lens = np.array([len(episode_batch)])
        if run_bc_online:
            dist_inputs = OnlineBCActionPredictor(
                {"trainer_config": trainer.config},
            ).predict_actions(input_dict)
        elif run == "bpd":
            dist_inputs = VIActionPredictor(cast(BPDPolicy, policy)).predict_actions(
                input_dict
            )
        else:
            dist_inputs, state_out = policy.model(input_dict, state_batches, seq_lens)
        assert policy.dist_class is not None
        dist_class: Type[TorchDistributionWrapper] = policy.dist_class
        action_dist = dist_class(cast(Any, dist_inputs), policy.model)
        actions = input_dict[SampleBatch.ACTIONS]
        episode_cross_entropies[episode_id] = -action_dist.logp(actions).mean().item()
        episode_accuracies[episode_id] = float(
            (action_dist.deterministic_sample() == actions).float().mean().item()
        )

        dist_inputs_no_stay = dist_inputs.clone()
        dist_inputs_no_stay[:, stay] = -np.inf
        action_dist_no_stay = dist_class(cast(Any, dist_inputs_no_stay), policy.model)
        episode_cross_entropies_no_stay[episode_id] = (
            -action_dist_no_stay.logp(actions)[actions != stay].mean().item()
        )
        episode_accuracies_no_stay[episode_id] = float(
            (action_dist_no_stay.deterministic_sample() == actions)[actions != stay]
            .float()
            .mean()
            .item()
        )

    results = {
        "cross_entropy": float(np.mean(list(episode_cross_entropies.values()))),
        "cross_entropy/episodes": episode_cross_entropies,
        "accuracy": float(np.mean(list(episode_accuracies.values()))),
        "accuracy/episodes": episode_accuracies,
        "no_stay/cross_entropy": float(
            np.mean(list(episode_cross_entropies_no_stay.values()))
        ),
        "no_stay/cross_entropy/episodes": episode_cross_entropies_no_stay,
        "no_stay/accuracy": float(np.mean(list(episode_accuracies_no_stay.values()))),
        "no_stay/accuracy/episodes": episode_accuracies_no_stay,
    }

    if out_path is not None:
        results_fname = f"{out_path}_prediction_results.json"
        _log.info(f"Saving results to {results_fname}")
        with open(results_fname, "w") as results_file:
            json.dump(results, results_file)

    return results
