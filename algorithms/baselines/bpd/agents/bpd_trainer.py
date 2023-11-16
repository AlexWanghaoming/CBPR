from typing import Tuple, Type
import numpy as np
import torch
from torch.distributions import Dirichlet

from ray.rllib.policy import Policy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator

from bpd.agents.bpd_policy import (
    BPDPolicy,
)


def project_probabilities(
    probabilities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project a tensor of action probabilities (categorical distributions over
    actions) onto a 2D polygon. Returns the resulting points and also
    the vertices of the polygon.
    """

    _, num_events = probabilities.size()

    event_thetas = (
        2 * np.pi * torch.arange(num_events, device=probabilities.device) / num_events
    )
    event_points = torch.stack(
        [
            torch.sin(event_thetas),
            torch.cos(event_thetas),
        ],
        dim=1,
    )
    probability_points = probabilities @ event_points
    return probability_points, event_points


class BPDMetrics:
    """
    Extra logging for the BPD implicit VI algorithm. Currently, it visualizes the
    distribution of policies at a particular state and compares it to the prior.
    """

    def __init__(self, workers: WorkerSet, num_samples=1000):
        self.workers = workers
        self.num_samples = num_samples

    def _add_metrics_for_policy(self, result, policy: Policy, policy_id: str) -> None:
        assert isinstance(policy, BPDPolicy)
        assert policy.model is not None

        obs = policy._sample_obs[None].repeat_interleave(self.num_samples, dim=0)
        policy_output = policy.model(
            {
                SampleBatch.OBS: policy.randomize_latent_in_obs(
                    obs,
                    action_space_size=policy.model.action_space.n,
                    all_unique=True,
                )
            }
        )[0].softmax(dim=1)
        prior_output = Dirichlet(
            torch.ones_like(policy_output) * policy.config["prior_concentration"]
        ).sample()

        policy_output_projected, vertices = project_probabilities(policy_output)
        prior_output_projected, _ = project_probabilities(prior_output)

        policy_output_projected = policy_output_projected.detach().cpu()
        prior_output_projected = prior_output_projected.detach().cpu()
        vertices = vertices.detach().cpu()

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(
            policy_output_projected[:, 0],
            policy_output_projected[:, 1],
            label="Policies",
            alpha=0.2,
        )
        ax.scatter(
            prior_output_projected[:, 0],
            prior_output_projected[:, 1],
            label="Prior",
            alpha=0.2,
        )
        vertices = torch.cat([vertices, vertices[:1]], dim=0)
        ax.plot(vertices[:, 0], vertices[:, 1], c="k")
        ax.legend()
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        dist_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        dist_image = dist_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        result[
            f"info/learner/{policy_id}/discriminator/distributions"
        ] = dist_image.transpose(2, 0, 1)[None, None]

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


class BPDTrainer(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return {
            **super().get_default_config(),
            # The temperature parameter for MaxEnt RL; the reward is effectively multiplied
            # by the reciprocal of this. Higher leads to more random behavior; 0 leads to
            # regular RL.
            "temperature": 1.0,
            # The concentration term for the Dirichlet prior over policies. Lower values
            # lead to more self-consistent policies, i.e. they almost always perform the
            # same action in the same state. Higher values lead to a distribution over
            # policies where actions are nearly independent across time, similar to regular
            # MaxEnt RL.
            "prior_concentration": 1.0,
            # The size (dimension) of the latent vector on which the policy is conditioned,
            # and whether the model also takes action probabilities from a random policy.
            "latent_size": 10,
        }

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return BPDPolicy
        else:
            raise NotImplementedError()

    @staticmethod
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        results = PPOTrainer.execution_plan(workers, config)
        return results.for_each(BPDMetrics(workers))
