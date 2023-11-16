"""
Distillation-prediction, where a distribution of policies is distilled through
behavior cloning into a single model with memory for predicting the next action given
past actions. This could be considered a meta-learned prediction model, or a
prediction model for a POMDP where the unobservable state is the human policy.
"""

import logging
import gym
import numpy as np
from gym import spaces
from ray.rllib.models.torch.attention_net import AttentionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.agents import Trainer
import torch
from torch.nn import functional as F
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Callable,
    cast,
)

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import (
    ParallelRollouts,
    ConcatBatches,
)
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    MultiAgentBatch,
    SampleBatch,
    DEFAULT_POLICY_ID,
)
from ray.rllib.utils.typing import (
    PolicyID,
    TrainerConfigDict,
    TensorType,
)
from ray.util.iter import LocalIterator
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_policy import LearningRateSchedule, TorchPolicy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    sequence_mask,
)
from bpd.agents.utils import get_select_experiences

from bpd.training_utils import load_policies_from_checkpoint

logger = logging.getLogger(__name__)

# Adds the following updates to the (base) `Trainer` config in
# rllib/agents/trainer.py (`COMMON_CONFIG` dict).
DEFAULT_CONFIG = with_common_config(
    {
        # Function which maps the ID of a policy acting in the environment to the
        # policy ID which should be trained to mimic it.
        "distillation_mapping_fn": None,
        # Optional checkpoint from which to load policy model weights to distill.
        "checkpoint_to_load_policies": None,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,
    }
)


def validate_config(config: TrainerConfigDict) -> None:
    # SGD minibatch size must be smaller than train_batch_size (b/c
    # we subsample a batch of `sgd_minibatch_size` from the train-batch for
    # each `sgd_num_iter`).
    if config["sgd_minibatch_size"] > config["train_batch_size"]:
        raise ValueError(
            "`sgd_minibatch_size` ({}) must be <= "
            "`train_batch_size` ({}).".format(
                config["sgd_minibatch_size"], config["train_batch_size"]
            )
        )

    # Multi-gpu not supported for PyTorch and tf-eager.
    if config["framework"] != "torch":
        raise ValueError("only PyTorch is supported")


class DistillationPredictionPolicy(TorchPolicy, LearningRateSchedule):
    # Cross-entropy during the first epoch of SGD. This is a more reliable metric
    # for prediction performance because the model has not yet been able to train on
    # the data.
    _initial_cross_entropy: List[float]

    def __init__(self, observation_space, action_space, config):
        config = Trainer.merge_trainer_configs(
            {**DEFAULT_CONFIG, "worker_index": None}, config
        )

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        self._initial_cross_entropy = []

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: TorchModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        # Replace initial state from teacher model with initial state from student model.
        for state_index, initial_state in enumerate(model.get_initial_state()):
            if isinstance(initial_state, np.ndarray):
                initial_state_tensor = torch.from_numpy(initial_state)
            else:
                initial_state_tensor = initial_state
            train_batch[f"state_in_{state_index}"] = initial_state_tensor[None].repeat(
                (len(train_batch[SampleBatch.SEQ_LENS]),)
                + (1,) * len(initial_state_tensor.size())
            )

        # If the model is a transformer, we need to add additional state to the batch.
        if isinstance(model, AttentionWrapper):
            for data_col, view_req in self.view_requirements.items():
                if data_col.startswith("state_in_"):
                    train_batch[data_col] = np.zeros(
                        (
                            len(train_batch["seq_lens"]),
                            view_req.shift_to - view_req.shift_from + 1,
                        )
                        + view_req.space.shape
                    )

        train_batch.set_training(True)
        logits, state = model(train_batch)
        action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch["seq_lens"])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch["seq_lens"], max_seq_len, time_major=model.is_time_major()
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # Compute cross entropy loss.
        cross_entropy = -action_dist.logp(train_batch[SampleBatch.ACTIONS])
        mean_cross_entropy = reduce_mean_valid(cross_entropy)

        total_loss = reduce_mean_valid(cross_entropy)

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_cross_entropy"] = mean_cross_entropy

        batches_per_epoch: int = (
            self.config["train_batch_size"] // self.config["sgd_minibatch_size"]
        )
        if len(self._initial_cross_entropy) < batches_per_epoch:
            self._initial_cross_entropy.append(mean_cross_entropy.item())
        self._last_batch = train_batch

        return total_loss

    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return cast(
            Dict[str, TensorType],
            convert_to_numpy(
                {
                    "cur_lr": self.cur_lr,
                    "total_loss": torch.mean(
                        torch.stack(
                            cast(List[torch.Tensor], self.get_tower_stats("total_loss"))
                        )
                    ),
                    "cross_entropy": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats("mean_cross_entropy"),
                            )
                        )
                    ),
                }
            ),
        )

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr


class MapExperiences:
    """
    Callable used to transfer experiences from policies acting in the environment
    to the policies that are being distilled onto.
    """

    def __init__(self, distillation_mapping_fn: Callable[[PolicyID], PolicyID]):
        self.distillation_mapping_fn = distillation_mapping_fn

    def __call__(self, samples: SampleBatch) -> MultiAgentBatch:
        multiagent_samples: MultiAgentBatch
        if isinstance(samples, MultiAgentBatch):
            multiagent_samples = samples
        else:
            multiagent_samples = MultiAgentBatch(
                {DEFAULT_POLICY_ID: samples}, len(samples)
            )

        multiagent_samples = MultiAgentBatch(
            {
                self.distillation_mapping_fn(policy_id): batch
                for policy_id, batch in multiagent_samples.policy_batches.items()
            },
            multiagent_samples.count,
        )

        return multiagent_samples


class ProcessExperiences:
    """
    Crop observations down to the size used by the actual policy being distilled.
    This hugely speeds up distillation of a latent-conditioned policy because it avoids
    copying all the latent vectors over to GPU.

    Also adds additional keys to the batches if necessary based on view requirements.
    """

    def __init__(
        self,
        policies: Dict[
            PolicyID,
            Tuple[Optional[Type[Policy]], gym.Space, gym.Space, Dict[str, Any]],
        ],
        workers: WorkerSet,
    ):
        self.policies = policies
        self.workers = workers

    def __call__(self, samples: MultiAgentBatch) -> MultiAgentBatch:
        new_policy_batches: Dict[PolicyID, SampleBatch] = dict(samples.policy_batches)
        batch: SampleBatch
        for policy_id, batch in samples.policy_batches.items():
            policy = cast(
                TorchPolicy, self.workers.local_worker().get_policy(policy_id)
            )
            model = policy.model
            assert isinstance(model, TorchModelV2)
            new_batch = batch.copy(shallow=True)

            if (
                new_batch.get(SampleBatch.SEQ_LENS) is None
                and model.get_initial_state()
            ):
                # Add seq_lens if it's needed and not there.
                max_seq_len = policy.config["model"]["max_seq_len"]
                if new_batch.count % max_seq_len != 0:
                    raise RuntimeError(
                        "batch does not divide evenly into sequences "
                        f"(count={new_batch.count}, max_seq_len={max_seq_len}"
                    )
                num_seqs = new_batch.count // max_seq_len
                new_batch[SampleBatch.SEQ_LENS] = np.array([max_seq_len] * num_seqs)

            initial_state = [
                convert_to_numpy(state) for state in model.get_initial_state()
            ]
            for key, view_requirement in policy.view_requirements.items():
                if key not in new_batch:
                    if key.startswith("state_in_"):
                        state_index = int(key[len("state_in_") :])
                        new_batch[key] = np.repeat(
                            initial_state[state_index][np.newaxis],
                            len(new_batch[SampleBatch.SEQ_LENS]),
                            axis=0,
                        )
                    elif key.startswith("state_out_"):
                        pass  # Don't actually need this for training generally.
                    elif key.startswith("prev_"):
                        assert view_requirement.shift == -1
                        prev = new_batch[view_requirement.data_col][:-1]
                        prev = np.concatenate([np.zeros_like(prev[:1]), prev], axis=0)
                        new_batch[key] = prev
                    elif key.startswith("seq_lens"):
                        # Should already be added.
                        assert new_batch[SampleBatch.SEQ_LENS] is not None
                    else:
                        raise RuntimeError(
                            f"unable to fulfill view requirement {key}={view_requirement}"
                        )

            _, obs_space, _, _ = self.policies[policy_id]
            if isinstance(obs_space, spaces.Box):
                obs_keys = [
                    key
                    for key, view_requirement in policy.view_requirements.items()
                    if view_requirement.data_col == SampleBatch.OBS
                    or key == SampleBatch.OBS
                ]
                for obs_key in obs_keys:
                    obs = new_batch[obs_key]
                    obs_shape = obs.shape[1:]
                    obs_space_shape = obs_space.shape
                    if len(obs_space_shape) > len(obs_shape):
                        # Observations must be flattened.
                        obs_space_shape = (np.prod(obs_space_shape),)
                    if any(
                        obs_dim > space_dim
                        for obs_dim, space_dim in zip(obs_shape, obs_space_shape)
                    ):
                        obs_slice = [slice(None)] + [
                            slice(0, space_dim) for space_dim in obs_space_shape
                        ]
                        new_obs = obs[tuple(obs_slice)]
                        new_batch[obs_key] = new_obs

                    # Unflatten observations if necessary.
                    obs = new_batch[obs_key]
                    if len(obs.shape[1:]) == 1 and len(obs_space.shape) > 1:
                        unflattened_obs = obs.reshape((obs.shape[0],) + obs_space.shape)
                        new_batch[obs_key] = unflattened_obs

            new_policy_batches[policy_id] = new_batch

        new_samples: MultiAgentBatch = MultiAgentBatch(
            new_policy_batches, samples.count
        )
        return new_samples


class PredictionMetrics:
    """
    Extra logging for the distillation-prediction trainer. It currently adds a plot
    of prediction performance over time.
    """

    def __init__(self, config: TrainerConfigDict, workers: WorkerSet):
        self.config = config
        self.workers = workers

    def _add_metrics_for_policy(
        self, result, policy: Policy, policy_id: PolicyID
    ) -> None:
        assert isinstance(policy, DistillationPredictionPolicy)

        result[f"info/learner/{policy_id}/initial_cross_entropy"] = np.mean(
            policy._initial_cross_entropy
        )
        policy._initial_cross_entropy = []

        if not policy.is_recurrent():
            return

        batch = policy._last_batch
        batch.set_training(False)
        assert policy.model is not None
        policy_output, _ = policy.model(batch)
        cross_entropy = F.cross_entropy(
            policy_output, batch[SampleBatch.ACTIONS], reduction="none"
        )

        seq_lens = batch["seq_lens"]
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = cross_entropy.shape[0] // seq_lens.shape[0]
        cross_entropy = add_time_dimension(
            cross_entropy,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=False,
        )

        bucket_size = 10
        cross_entropy_bucketed = cross_entropy.reshape(
            (
                cross_entropy.size()[0],
                cross_entropy.size()[1] // bucket_size,
                bucket_size,
            )
        )
        bucket_cross_entropies = cross_entropy_bucketed.mean(dim=2)
        mean_cross_entropies = bucket_cross_entropies.mean(dim=0).detach().cpu()
        std_cross_entropies = bucket_cross_entropies.std(dim=0).detach().cpu()
        t = np.arange(len(mean_cross_entropies)) * bucket_size

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.fill_between(
            t,
            mean_cross_entropies - std_cross_entropies,
            mean_cross_entropies + std_cross_entropies,
            alpha=0.1,
        )
        ax.plot(t, mean_cross_entropies)
        ax.plot(t, np.ones_like(t) * np.log(policy_output.shape[-1]), c="k", ls="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cross-entropy")
        fig.tight_layout()
        fig.canvas.draw()
        plot_image = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype=np.uint8,
        )
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        result[f"info/learner/{policy_id}/cross_entropy_plot"] = plot_image.transpose(
            2, 0, 1
        )[None, None]

    def __call__(self, result):
        if hasattr(self.workers, "foreach_policy_to_train"):
            self.workers.local_worker().foreach_policy_to_train(  # type: ignore
                lambda policy, policy_id, **kwargs: self._add_metrics_for_policy(
                    result, policy, policy_id
                )
            )
        else:
            # For RLlib versions < 1.12
            self.workers.local_worker().foreach_trainable_policy(
                lambda policy, policy_id, **kwargs: self._add_metrics_for_policy(
                    result, policy, policy_id
                )
            )
        return result


def execution_plan(
    workers: WorkerSet, config: TrainerConfigDict
) -> LocalIterator[dict]:
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Map batches to the policies getting distilled onto.
    rollouts_multiagent = rollouts.for_each(
        MapExperiences(config["distillation_mapping_fn"])
    ).for_each(ProcessExperiences(config["multiagent"]["policies"], workers))

    # Collect batches for the trainable policies.
    selected_rollouts = rollouts_multiagent.for_each(get_select_experiences(workers))
    # Concatenate the SampleBatches into one.
    combined_rollouts = selected_rollouts.combine(
        ConcatBatches(
            min_batch_size=config["train_batch_size"],
            count_steps_by=config["multiagent"]["count_steps_by"],
        )
    )

    train_op = combined_rollouts.for_each(
        TrainOneStep(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
        )
    )

    # Return training metrics.
    results = StandardMetricsReporting(train_op, workers, config)
    return results.for_each(PredictionMetrics(config, workers))


def after_init(trainer):
    # Add view requirements for student policies that might not already be required
    # by the teacher policies.
    workers: WorkerSet = trainer.workers
    all_view_requirements = dict(
        workers.foreach_policy(
            lambda policy, policy_id: (policy_id, policy.view_requirements)
        )
    )
    distillation_mapping_fn: Callable[[PolicyID], PolicyID] = trainer.config[
        "distillation_mapping_fn"
    ]
    distillation_view_requirements = {
        policy_id: all_view_requirements[distillation_mapping_fn(policy_id)]
        for policy_id in all_view_requirements.keys()
        if distillation_mapping_fn(policy_id) in all_view_requirements
    }

    def add_trajectory_views(policy: Policy, policy_id: PolicyID):
        if policy_id in distillation_view_requirements:
            for data_col, view_requirement in distillation_view_requirements[
                policy_id
            ].items():
                if data_col not in policy.view_requirements:
                    policy.view_requirements[data_col] = view_requirement
                model: ModelV2 = cast(Any, policy).model
                if data_col not in model.view_requirements:
                    model.view_requirements[data_col] = view_requirement

    cast(WorkerSet, trainer.workers).foreach_policy(add_trajectory_views)

    if trainer.config["checkpoint_to_load_policies"] is not None:
        load_policies_from_checkpoint(
            trainer.config["checkpoint_to_load_policies"], trainer
        )


DistillationPredictionTrainer = build_trainer(
    name="DistillationPredictionTrainer",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=DistillationPredictionPolicy,
    get_policy_class=lambda config: DistillationPredictionPolicy,
    after_init=after_init,
    execution_plan=execution_plan,
)
