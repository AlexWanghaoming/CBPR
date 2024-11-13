from typing import Dict, List, Union, Type, cast
import torch
import random
from torch.nn import functional as F

from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import (
    ConcatBatches,
    ParallelRollouts,
    StandardizeFields,
)
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.policy.policy import Policy
from ray.util.iter import LocalIterator
from ray.rllib.agents.ppo.ppo_torch_policy import (
    PPOTorchPolicy,
)
from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID,
    MultiAgentBatch,
    SampleBatch,
)
from ray.rllib.policy.torch_policy import (
    EntropyCoeffSchedule,
    LearningRateSchedule,
    TorchPolicy,
)
from ray.rllib.utils.typing import (
    TensorType,
    TrainerConfigDict,
    SampleBatchType,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.agents.ppo.ppo import (
    PPOTrainer,
    UpdateKL,
    warn_about_bad_reward_scales,
)
from ray.rllib.utils.numpy import convert_to_numpy

from bpd.agents.utils import get_select_experiences

from .bpd_policy import ModelWithDiscriminator


class DemonstrationMixin(object):
    all_demonstration_episodes: List[SampleBatch]

    def __init__(
        self,
        demonstration_input: str,
        demonstration_num_episodes: int = 1,
    ):
        demonstration_reader = JsonReader(demonstration_input)
        self.demonstration_num_episodes = demonstration_num_episodes

        self.all_demonstration_episodes = []
        batches_fnames: List[str] = demonstration_reader.files
        for batches_fname in batches_fnames:
            with open(batches_fname, "r") as batches_file:
                for line in batches_file:
                    line = line.strip()
                    if line:
                        batch = demonstration_reader._from_json(line)
                        assert isinstance(batch, SampleBatch)
                        self.all_demonstration_episodes.extend(batch.split_by_episode())

    def get_demonstration_batch(self) -> SampleBatch:
        episodes = random.sample(
            self.all_demonstration_episodes, self.demonstration_num_episodes
        )
        return cast(
            SampleBatch,
            cast(TorchPolicy, self)._lazy_tensor_dict(
                cast(SampleBatch, SampleBatch.concat_samples(episodes))
            ),
        )


class GailPolicyType(TorchPolicy, DemonstrationMixin):
    """Dummy class used for type checking."""

    model: TorchModelV2

    # Metrics saved from loss function:
    _discriminator_loss: torch.Tensor
    _discriminator_policy_score: torch.Tensor
    _discriminator_demonstration_score: torch.Tensor
    _discriminator_reward: torch.Tensor
    _demonstration_cross_entropy: torch.Tensor


class GailPolicy(PPOTorchPolicy, DemonstrationMixin):
    def __init__(self, observation_space, action_space, config):
        config = Trainer.merge_trainer_configs(
            {
                **GailTrainer.get_default_config(),
                "worker_index": None,
            },
            config,
        )

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        DemonstrationMixin.__init__(
            self, config["demonstration_input"], config["demonstration_num_episodes"]
        )

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, ModelWithDiscriminator)

        # Train discriminator.
        discriminator_policy_scores = model.discriminator(train_batch)

        demonstration_batch = self._lazy_tensor_dict(self.get_demonstration_batch())
        discriminator_demonstration_scores = model.discriminator(demonstration_batch)

        discriminator_loss = (
            F.softplus(discriminator_demonstration_scores).mean()
            + F.softplus(-discriminator_policy_scores).mean()
        )

        # Store additional stats in policy for stats_fn.
        model.tower_stats["discriminator_loss"] = discriminator_loss
        model.tower_stats[
            "discriminator_policy_score"
        ] = discriminator_policy_scores.mean()
        model.tower_stats[
            "discriminator_demonstration_score"
        ] = discriminator_demonstration_scores.mean()

        demonstration_model_out, _ = model.from_batch(demonstration_batch)
        demonstration_action_dist = dist_class(demonstration_model_out, model)
        model.tower_stats[
            "demonstration_cross_entropy"
        ] = -demonstration_action_dist.logp(
            demonstration_batch[SampleBatch.ACTIONS]
        ).mean()

        model.tower_stats["discriminator_reward"] = train_batch[
            SampleBatch.REWARDS
        ].mean()

        ppo_loss = super().loss(
            model,
            dist_class,
            train_batch,
        )

        loss = ppo_loss + discriminator_loss
        return loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return cast(
            Dict[str, TensorType],
            convert_to_numpy(
                {
                    **super().extra_grad_info(train_batch),
                    "discriminator/loss": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats("discriminator_loss"),
                            )
                        )
                    ),
                    "discriminator/policy_score": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats("discriminator_policy_score"),
                            )
                        )
                    ),
                    "discriminator/demonstration_score": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats(
                                    "discriminator_demonstration_score"
                                ),
                            )
                        )
                    ),
                    "discriminator/reward": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats("discriminator_reward"),
                            )
                        )
                    ),
                    "demonstration_cross_entropy": torch.mean(
                        torch.stack(
                            cast(
                                List[torch.Tensor],
                                self.get_tower_stats("demonstration_cross_entropy"),
                            )
                        )
                    ),
                }
            ),
        )


class DiscriminatorReward:
    """
    Callable that replaces rewards in rollouts with output of the discriminator.
    """

    def __init__(self, workers: WorkerSet):
        self.workers = workers

    def __call__(
        self, samples: Union[MultiAgentBatch, SampleBatch]
    ) -> Union[MultiAgentBatch, SampleBatch]:
        wrapped = False

        if isinstance(samples, SampleBatch):
            multiagent_batch = MultiAgentBatch(
                {DEFAULT_POLICY_ID: samples}, samples.count
            )
            wrapped = True
        else:
            multiagent_batch = samples

        def replace_reward_for_policy(policy: Policy, policy_id: str):
            policy = cast(GailPolicyType, policy)
            batch = multiagent_batch.policy_batches[policy_id]
            model = cast(ModelWithDiscriminator, policy.model)
            discriminator_policy_scores = model.discriminator(
                policy._lazy_tensor_dict(batch.copy())
            )
            batch[SampleBatch.REWARDS] = (
                F.softplus(-discriminator_policy_scores)[:, 0].cpu().detach().numpy()
            )
            # Need to recompute advantages.
            multiagent_batch.policy_batches[policy_id] = compute_gae_for_sample_batch(
                policy,
                batch,
            )

        self.workers.foreach_policy(replace_reward_for_policy)

        if wrapped:
            samples = multiagent_batch.policy_batches[DEFAULT_POLICY_ID]
        else:
            samples = multiagent_batch

        return samples


class GailTrainer(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return {
            **super().get_default_config(),
            # Directory with offline-format data to use as demonstrations.
            "demonstration_input": None,
            # How many episodes of demonstration data to use per SGD step.
            "demonstration_num_episodes": 1,
        }

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return GailPolicy
        else:
            raise NotImplementedError()

    @staticmethod
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        rollouts = cast(
            LocalIterator[SampleBatchType], ParallelRollouts(workers, mode="bulk_sync")
        )

        # Collect batches for the trainable policies.
        rollouts = rollouts.for_each(get_select_experiences(workers))

        # Replace reward using discriminator.
        rollouts = rollouts.for_each(DiscriminatorReward(workers))

        # Concatenate the SampleBatches into one.
        rollouts = rollouts.combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            )
        )
        # Standardize advantages.
        rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

        # Perform one training step on the combined + standardized batch.
        if config["simple_optimizer"]:
            train_op = rollouts.for_each(
                TrainOneStep(
                    workers,
                    num_sgd_iter=config["num_sgd_iter"],
                    sgd_minibatch_size=config["sgd_minibatch_size"],
                )
            )
        else:
            train_op = rollouts.for_each(
                TrainTFMultiGPU(
                    workers=workers,
                    sgd_minibatch_size=config["sgd_minibatch_size"],
                    num_sgd_iter=config["num_sgd_iter"],
                    num_gpus=config["num_gpus"],
                    shuffle_sequences=config["shuffle_sequences"],
                    _fake_gpus=config["_fake_gpus"],
                    framework=config.get("framework", "tf"),
                )
            )

        # Update KL after each round of training.
        train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))  # type: ignore

        # Warn about bad reward scales and return training metrics.
        return StandardMetricsReporting(train_op, workers, config).for_each(
            lambda result: warn_about_bad_reward_scales(config, result)  # type: ignore
        )
