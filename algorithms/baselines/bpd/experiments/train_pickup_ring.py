from typing import Optional
from logging import Logger
import os
import torch
import numpy as np

import ray
from ray.rllib.utils.typing import TrainerConfigDict

from sacred import Experiment
from sacred import SETTINGS as sacred_settings

from bpd.training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
)
from bpd.agents.bpd_trainer import BPDTrainer

ex = Experiment("train_pickup_ring")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


@ex.config
def sacred_config(_log):  # noqa
    # Environment
    ring_size = 8
    horizon = 100

    # Training
    num_workers = 2
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    train_batch_size = 2000
    sgd_minibatch_size = 2000
    rollout_fragment_length = horizon
    num_training_iters = 500  # noqa: F841
    lr = 1e-3
    grad_clip = 0.1
    gamma = 0.9
    gae_lambda = 0.98
    vf_loss_coeff = 1e-2
    entropy_coeff_start = 0
    entropy_coeff_end = 0
    entropy_coeff_horizon = 2e5
    kl_coeff = 0.2
    clip_param = 0.05
    num_sgd_iter = 8

    # Model
    size_hidden_layers = 64
    latent_size = 2
    max_seq_len = 10
    discriminate_sequences = False
    model_config = {
        "custom_model": "pickup_ring_distribution_model",
        "max_seq_len": max_seq_len,
        "vf_share_layers": True,
        "custom_model_config": {
            "latent_size": latent_size,
            "discriminate_sequences": discriminate_sequences,
        },
        "fcnet_hiddens": [size_hidden_layers] * 3,
    }

    # BPD
    temperature = 1.0
    prior_concentration = 1.0

    latent_dist = lambda latent_size=latent_size: np.random.normal(0, 1, latent_size)
    checkpoint_to_load_policies = None  # noqa: F841
    latents_per_iteration = 4
    episodes_per_latent = (train_batch_size // latents_per_iteration) // horizon
    if (
        episodes_per_latent * latents_per_iteration * horizon != train_batch_size
        or num_workers * (latents_per_iteration // max(num_workers, 1))
        != latents_per_iteration
    ):
        _log.warning("Uneven split of episodes or workers across latent vectors!")

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs/pickup_ring"  # noqa: F841
    experiment_tag = None
    experiment_name_parts = [
        f"ring_size_{ring_size}",
        f"temperature_{temperature}",
        f"concentration_{prior_concentration}",
    ]
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
    checkpoint_path = None  # noqa: F841

    env_id = "latent_wrapper"
    env_config = {
        "env": "pickup_ring",
        "env_config": {
            "ring_size": ring_size,
            "horizon": horizon,
        },
        "latent_dist": latent_dist,
        "episodes_per_latent": episodes_per_latent,
        "agents_with_latent": {0},
    }

    config: TrainerConfigDict = {  # noqa: F841
        "env": env_id,
        "env_config": env_config,
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
        "seed": seed,
        "model": model_config,
        "entropy_coeff_schedule": [
            (0, entropy_coeff_start),
            (entropy_coeff_horizon, entropy_coeff_end),
        ],
        "framework": "torch",
        "temperature": temperature,
        "prior_concentration": prior_concentration,
        "latent_size": latent_size,
    }


@ex.automain
def main(
    config,
    log_dir,
    experiment_name,
    num_training_iters,
    save_freq,
    checkpoint_to_load_policies: Optional[str],
    checkpoint_path: Optional[str],
    _log: Logger,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer = BPDTrainer(
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
