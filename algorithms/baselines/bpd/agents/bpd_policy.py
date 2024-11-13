from typing import Dict, List, Optional, Tuple, Union, Type, cast
from ray.rllib.models.modelv2 import restore_original_dimensions
import torch
import random
from torch.nn import functional as F
from torch.distributions import Dirichlet, Distribution

from ray.rllib.models import ModelV2
from ray.rllib.agents import Trainer
from ray.rllib.agents.ppo.ppo_torch_policy import (
    PPOTorchPolicy,
)
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import (
    EntropyCoeffSchedule,
    LearningRateSchedule,
    TorchPolicy,
)
from ray.rllib.utils.typing import ModelInputDict, TensorType
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.numpy import convert_to_numpy


class ModelWithDiscriminator(TorchModelV2):
    def discriminator(
        self,
        input_dict: ModelInputDict,
        seq_lens: Optional[torch.Tensor] = None,
        detached: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError()


class BPDPolicy(PPOTorchPolicy):
    """
    Policy that implements the Boltzmann Policy Distribution using implicit
    variational inference.
    """

    def __init__(self, observation_space, action_space, config):
        from .bpd_trainer import BPDTrainer

        config = Trainer.merge_trainer_configs(
            {**BPDTrainer.get_default_config(), "worker_index": None},
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

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

        # Set beta1 in Adam to 0.5 to stabilize GAN-type training.
        self._optimizers[0].param_groups[0]["betas"] = (0.5, 0.999)
        self._initialize_loss_from_dummy_batch()


    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, TorchModelV2)

        # if hasattr(policy.model, "extra_init"):
        #     policy.model.extra_init(train_batch)  # type: ignore

        loss = super().loss(
            model,
            dist_class,
            train_batch,
        )

        if hasattr(model, "logits"):
            model_out = model.logits()
        else:
            model_out, _ = model(train_batch)

        prior_concentration: float = self.config["prior_concentration"]
        prior: Distribution

        obs = cast(
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            restore_original_dimensions(
                train_batch[SampleBatch.OBS], model.obs_space, "torch"
            ),
        )

        model = cast(ModelWithDiscriminator, model)

        # Approximate KL from prior with discriminator.
        prior_kl_div = model.discriminator(
            {
                SampleBatch.OBS: obs,
                SampleBatch.ACTION_PROB: model_out.softmax(dim=1),
            },
            seq_lens=train_batch.get("seq_lens"),
            detached=True,
        )
        prior_kl_div = prior_kl_div * (prior_kl_div.size()[0] / model_out.size()[0])
        loss += self.config["temperature"] * prior_kl_div.mean()

        # Train discriminator.
        with torch.no_grad():
            discriminator_model_out, _ = model(
                {
                    SampleBatch.OBS: self.randomize_latent_in_obs(
                        obs,
                        model.action_space.n,
                    ),
                }
            )

        discriminator_policy_scores = model.discriminator(
            {
                SampleBatch.OBS: obs,
                SampleBatch.ACTION_PROB: discriminator_model_out.softmax(
                    dim=1
                ).detach(),
            },
            seq_lens=train_batch.get("seq_lens"),
        )

        prior = Dirichlet(torch.ones_like(model_out) * prior_concentration)
        discriminator_prior_scores = model.discriminator(
            {
                SampleBatch.OBS: obs,
                SampleBatch.ACTION_PROB: BPDPolicy.sample_from_policy_prior(
                    prior, train_batch
                ),
            },
            seq_lens=train_batch.get("seq_lens"),
        )
        discriminator_loss = (
            F.softplus(discriminator_prior_scores)
            + F.softplus(-discriminator_policy_scores)
        ).mean()
        loss += discriminator_loss.mean()

        # Store additional stats in policy for stats_fn.
        model.tower_stats["prior_kl_div"] = prior_kl_div.mean()
        model.tower_stats["elbo"] = (train_batch[SampleBatch.REWARDS].float().mean() - self.config["temperature"] * prior_kl_div.mean())
        model.tower_stats["discriminator_loss"] = discriminator_loss
        model.tower_stats[
            "discriminator_policy_score"
        ] = discriminator_policy_scores.mean()
        model.tower_stats[
            "discriminator_prior_score"
        ] = discriminator_prior_scores.mean()

        self._sample_obs = train_batch[SampleBatch.OBS][
            random.randrange(0, train_batch[SampleBatch.OBS].size()[0])
        ]

        return loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().extra_grad_info(train_batch)
        stats.update(
            {
                "prior_kl_div": torch.mean(
                    torch.stack(
                        cast(List[torch.Tensor], self.get_tower_stats("prior_kl_div"))
                    )
                ),
                "elbo": torch.mean(
                    torch.stack(cast(List[torch.Tensor], self.get_tower_stats("elbo")))
                ),
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
                "discriminator/prior_score": torch.mean(
                    torch.stack(
                        cast(
                            List[torch.Tensor],
                            self.get_tower_stats("discriminator_prior_score"),
                        )
                    )
                ),
            }
        )

        return cast(Dict[str, TensorType], convert_to_numpy(stats))

    @staticmethod
    def sample_from_policy_prior(
        prior: Distribution,
        train_batch: SampleBatch,
    ) -> TensorType:
        """
        Sample from a prior distribution over policies, ensuring for sequences that
        identical observations lead to identical action probabilities.
        """

        seq_lens: Optional[torch.Tensor] = train_batch.get("seq_lens")
        if seq_lens is None:
            return prior.sample()
        else:
            max_seq_len = train_batch[SampleBatch.OBS].shape[0] // seq_lens.shape[0]
            obs = add_time_dimension(
                train_batch[SampleBatch.OBS],
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=False,
            )
            flat_obs = obs.flatten(start_dim=2)
            identical_obs = torch.all(flat_obs[:, None] == flat_obs[:, :, None], dim=3)
            unique_obs_id = identical_obs.long().argmax(dim=1)
            prior_sample = prior.sample()
            prior_sample_w_time = add_time_dimension(
                prior.sample(),
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=False,
            )
            prior_sample_w_time = torch.gather(
                prior_sample_w_time,
                1,
                unique_obs_id.unsqueeze(-1).expand(prior_sample_w_time.size()),
            )
            return prior_sample_w_time.reshape(prior_sample.size())

    def randomize_latent_in_obs(
        self,
        obs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        action_space_size: int = 0,
        all_unique: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Replaces the latent vectors (and random policy action probabilities) in the given
        observations with newly sampled ones. The structure of which latent vectors and
        action probabilities are the same is maintained in the newly randomized
        observations, unless all_unique=True.
        """

        latent_size = self.config["latent_size"]
        total_latent_size = latent_size

        if isinstance(obs, (tuple, list)):
            _, latents = obs
        else:
            latents_obs = obs
            if len(latents_obs.size()) > 3:
                latents_obs = latents_obs.flatten(start_dim=1, end_dim=-2)
            elif len(latents_obs.size()) == 2:
                latents_obs = latents_obs[:, None, :]
            latents = latents_obs[:, 0, -total_latent_size:]

        latents_for_comparison = latents[:, :10]
        identical_latents = torch.all(
            latents_for_comparison[None] == latents_for_comparison[:, None], dim=2
        )
        unique_latents_id = identical_latents.long().argmax(dim=1)
        new_latents = torch.normal(torch.zeros_like(latents))
        if not all_unique:
            new_latents = torch.gather(
                new_latents, 0, unique_latents_id.unsqueeze(-1).expand(latents.size())
            )

        if isinstance(obs, (tuple, list)):
            return obs[0], new_latents
        else:
            while len(new_latents.size()) < len(obs.size()):
                new_latents = new_latents[:, None]

            new_obs = obs.clone()
            new_obs[..., -total_latent_size:] = new_latents
            return new_obs
