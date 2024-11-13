from typing import Dict, List, Optional, Set, Type, cast
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.utils.numpy import convert_to_numpy

import torch
from torch.optim import Optimizer
import random
import logging


logger = logging.getLogger(__name__)


class LearningRateDrop(object):
    """Mixin for TorchPolicy that automatically drops the learning rate."""

    loss_history: List[float]

    def __init__(self, lr: float, patience: int, factor=0.1, delta=1e-4):
        self.loss_history = []
        self.cur_lr = lr
        self.lr_drop_patience = patience
        self.lr_drop_factor = factor
        self.lr_drop_delta = delta

    def update_lr(self, optimizers: List[Optimizer]):
        if len(self.loss_history) > self.lr_drop_patience + 1:
            any_improvement = False
            for prev_loss, new_loss in zip(
                self.loss_history[-self.lr_drop_patience - 1 : -1],
                self.loss_history[-self.lr_drop_patience :],
            ):
                if prev_loss - new_loss >= self.lr_drop_delta:
                    any_improvement = True
            if not any_improvement:
                self.cur_lr *= self.lr_drop_factor

        for opt in optimizers:
            for p in opt.param_groups:
                p["lr"] = self.cur_lr


class BCTorchPolicy(TorchPolicy, LearningRateDrop):
    def __init__(self, observation_space, action_space, config):
        config = Trainer.merge_trainer_configs(
            {**BCTrainer.get_default_config(), "worker_index": None}, config
        )

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        LearningRateDrop.__init__(self, config["lr"], config["lr_drop_patience"])

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, TorchModelV2)

        episode_ids: Set[int] = set(train_batch[SampleBatch.EPS_ID].tolist())
        episode_in_validation: Dict[int, bool] = {
            episode_id: random.Random(episode_id).random()
            < self.config["validation_prop"]
            for episode_id in episode_ids
        }
        validation_mask = torch.tensor(
            [
                episode_in_validation[episode_id.item()]
                for episode_id in train_batch[SampleBatch.EPS_ID]
            ],
            dtype=torch.bool,
            device=self.device,
        )

        model_out, _ = model.from_batch(train_batch)
        action_dist: ActionDistribution = dist_class(model_out, model)
        actions = train_batch[SampleBatch.ACTIONS]
        logprobs = action_dist.logp(actions)

        bc_loss = -torch.mean(logprobs[~validation_mask])
        model.tower_stats["bc_loss"] = bc_loss
        model.tower_stats["accuracy"] = (
            (action_dist.deterministic_sample() == actions)[~validation_mask]
            .float()
            .mean()
        )

        validation_cross_entropy: Optional[torch.Tensor]
        if torch.any(validation_mask):
            validation_cross_entropy = -logprobs[validation_mask].mean()
            model.tower_stats["validation_cross_entropy"] = validation_cross_entropy
        else:
            validation_cross_entropy = None
            model.tower_stats["validation_cross_entropy"] = torch.zeros(size=(0,))

        if self.config["validation_prop"] > 0:
            if validation_cross_entropy is not None:
                self.loss_history.append(validation_cross_entropy.item())
        else:
            self.loss_history.append(bc_loss.item())

        return bc_loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = {
            "bc_loss": torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats("bc_loss")))
            ),
            "accuracy": torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats("accuracy")))
            ),
            "cur_lr": self.cur_lr,
        }
        if self.get_tower_stats("validation_cross_entropy")[0] is not None:
            stats["validation/cross_entropy"] = torch.mean(
                torch.stack(
                    cast(
                        List[torch.Tensor],
                        self.get_tower_stats("validation_cross_entropy"),
                    )
                )
            )
        return cast(Dict[str, TensorType], convert_to_numpy(stats))

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        self.update_lr(self._optimizers)


class BCTrainer(Trainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return with_common_config(
            {
                # You should override this to point to an offline dataset (see agent.py).
                "input": "sampler",
                # No reward estimation.
                "input_evaluation": [],
                # If specified, clip the global norm of gradients by this amount.
                "grad_clip": None,
                # Whether to rollout "complete_episodes" or "truncate_episodes".
                "batch_mode": "complete_episodes",
                # Learning rate for adam optimizer.
                "lr": 1e-4,
                # If this is set to a number, then the learning rate will be dropped by
                # a factor of 10 after no improvement for this number of epochs.
                "lr_drop_patience": None,
                # Number of timesteps collected for each SGD round.
                "train_batch_size": 2000,
                # State/action pairs in each SGD minibatch.
                "sgd_minibatch_size": 100,
                # === Parallelism ===
                "num_workers": 0,
                # Use this proportion of the episodes for validation.
                "validation_prop": 0,
            }
        )

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return BCTorchPolicy
        else:
            raise NotImplementedError()

    @staticmethod
    def execution_plan(workers, config):
        rollouts = ParallelRollouts(workers, mode="bulk_sync")

        train_op = rollouts.for_each(
            TrainOneStep(workers, sgd_minibatch_size=config["sgd_minibatch_size"])
        )

        return StandardMetricsReporting(train_op, workers, config)
