from typing import cast
from typing_extensions import TypedDict
from gym import spaces
import torch
import tqdm
from torch.optim import Adam
from torch.distributions import Normal, kl_divergence

from ray.rllib.policy.sample_batch import SampleBatch

from .agents.bpd_policy import BPDPolicy


class VIActionPredictorConfig(TypedDict):
    lr: float
    sgd_iters_per_timestep: int
    elbo_samples: int
    prediction_samples: int


DEFAULT_CONFIG: VIActionPredictorConfig = {
    "lr": 1e-1,
    "sgd_iters_per_timestep": 1,
    "elbo_samples": 10,
    "prediction_samples": 1000,
}


class VIActionPredictor(object):
    """
    Does an explicit Bayes filter over the latent space of a conditional policy to
    predict future actions given past actions.
    """

    def __init__(self, policy: BPDPolicy, config={}):
        self.policy = policy
        self.latent_size: int = self.policy.config["latent_size"]
        self.config = DEFAULT_CONFIG
        self.config.update(config)

    def _forward_with_latents(
        self, episode_batch: SampleBatch, latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Given an episode sample batch with T timesteps and several latents of size NxL,
        this returns logits of size NxTxA (A is number of actions).
        """

        obs = episode_batch[SampleBatch.OBS]
        assert self.policy.model is not None
        if isinstance(self.policy.model.obs_space, spaces.Tuple):
            obs = obs.flatten(start_dim=1)
        repeated_obs = obs.repeat(latents.size()[0], *([1] * (len(obs.size()) - 1)))
        latents_to_append = latents.repeat_interleave(obs.size()[0], dim=0)
        while len(latents_to_append.size()) < len(repeated_obs.size()):
            latents_to_append = latents_to_append.unsqueeze(1)
        latents_to_append = latents_to_append.expand(
            *repeated_obs.size()[:-1],
            latents_to_append.size()[-1],
        )
        obs_with_latents = torch.cat([repeated_obs, latents_to_append], dim=-1)

        logits, _ = self.policy.model({SampleBatch.OBS: obs_with_latents})
        return cast(
            torch.Tensor,
            logits.reshape(
                latents.size()[0], obs.size()[0], self.policy.model.action_space.n
            ),
        )

    def predict_actions(self, episode_batch: SampleBatch) -> torch.Tensor:
        """
        Predict actions over the given episode, returning logits.
        """

        episode_batch = self.policy._lazy_tensor_dict(episode_batch)

        latents_mean = torch.zeros(self.latent_size, device=self.policy.device)
        latents_logstd = torch.zeros_like(latents_mean)
        latents_mean.requires_grad = True
        latents_logstd.requires_grad = True
        optim = Adam([latents_mean, latents_logstd], lr=self.config["lr"])

        latents_prior = Normal(
            torch.zeros_like(latents_mean),
            torch.ones_like(latents_logstd),
        )

        assert self.policy.model is not None
        episode_logits = torch.zeros(
            (episode_batch.count, self.policy.model.action_space.n),
            device=self.policy.device,
        )

        with tqdm.trange(
            episode_batch.count, position=0, desc="", leave=True
        ) as episode_iter:
            for t in episode_iter:
                # Update posterior.
                if t != 0:
                    num_sgd_iter = self.config["sgd_iters_per_timestep"]
                    for sgd_iter in range(num_sgd_iter):
                        latents_posterior = Normal(latents_mean, latents_logstd.exp())
                        elbo_latents = latents_posterior.rsample(
                            (self.config["elbo_samples"],)
                        )
                        logits = self._forward_with_latents(
                            self.policy._lazy_tensor_dict(episode_batch.slice(0, t)),
                            elbo_latents,
                        )
                        log_probs = logits - logits.logsumexp(2, keepdim=True)
                        log_likelihood = (
                            log_probs[
                                :,
                                torch.arange(t),
                                episode_batch[SampleBatch.ACTIONS][:t],
                            ]
                            .mean(dim=0)
                            .sum(dim=0)
                        )
                        prior_kl = kl_divergence(
                            latents_posterior,
                            latents_prior,
                        ).sum()
                        elbo = log_likelihood - prior_kl

                        import torch.nn.functional as F

                        ce = F.cross_entropy(
                            episode_logits[:t], episode_batch[SampleBatch.ACTIONS][:t]
                        )
                        episode_iter.set_description(
                            "  ".join(
                                [
                                    f"sgd {sgd_iter}/{num_sgd_iter}",
                                    f"elbo = {elbo.item():.2f}",
                                    f"ce_past = {(-log_likelihood / t).item():.2f}",
                                    f"ce = {ce.item():.2f}",
                                ]
                            )
                        )

                        optim.zero_grad()
                        (-elbo).backward()
                        optim.step()

                # Predict based on posterior.
                latents_posterior = Normal(latents_mean, latents_logstd.exp())
                prediction_latents = latents_posterior.sample(
                    (self.config["prediction_samples"],)
                )
                logits = self._forward_with_latents(
                    self.policy._lazy_tensor_dict(episode_batch.slice(t, t + 1)),
                    prediction_latents,
                )
                episode_logits[t] = logits[:, 0].logsumexp(dim=0).detach()

        return episode_logits
