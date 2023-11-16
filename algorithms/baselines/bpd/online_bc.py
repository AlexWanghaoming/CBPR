import functools
from typing import TypedDict
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import torch
import tqdm
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.models import ModelCatalog


class OnlineBCConfig(TypedDict):
    lr: float
    sgd_iters_per_timestep: int
    trainer_config: dict


DEFAULT_CONFIG: OnlineBCConfig = {
    "lr": 1e-3,
    "sgd_iters_per_timestep": 1,
    "trainer_config": {},
}


class OnlineBCActionPredictor(object):
    """
    Runs BC online, updating the learned human policy at each timestep with the data
    seen so far.
    """

    def __init__(self, config={}):
        self.config = DEFAULT_CONFIG
        self.config.update(config)

    def _forward(self, batch: SampleBatch, device: torch.device):
        batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=device)
        )
        logits, state = self.model(batch)
        return logits

    def predict_actions(self, episode_batch: SampleBatch) -> torch.Tensor:
        """
        Predict actions over the given episode, returning logits.
        """

        _, obs_space, action_space, policy_config = self.config["trainer_config"][
            "multiagent"
        ]["policies"][DEFAULT_POLICY_ID]

        model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            num_outputs=action_space.n,
            model_config=policy_config["model"],
            framework="torch",
        )
        assert isinstance(model, nn.Module)
        self.model = model

        optim = Adam(model.parameters(), lr=self.config["lr"])

        episode_logits = torch.zeros(
            (episode_batch.count, action_space.n),
        )

        if torch.cuda.is_available():
            model.cuda()
            episode_logits = episode_logits.cuda()
        device = episode_logits.device

        with tqdm.trange(
            episode_batch.count, position=0, desc="", leave=True
        ) as episode_iter:
            for t in episode_iter:
                # Update model.
                if t != 0:
                    num_sgd_iter = self.config["sgd_iters_per_timestep"]
                    for sgd_iter in range(num_sgd_iter):
                        logits = self._forward(episode_batch.slice(0, t), device)
                        log_probs = logits - logits.logsumexp(1, keepdim=True)
                        log_likelihood = log_probs[
                            torch.arange(t),
                            episode_batch[SampleBatch.ACTIONS][:t],
                        ].mean(dim=0)

                        ce = F.cross_entropy(
                            episode_logits[:t], episode_batch[SampleBatch.ACTIONS][:t]
                        )
                        episode_iter.set_description(
                            "  ".join(
                                [
                                    f"sgd {sgd_iter}/{num_sgd_iter}",
                                    f"ce_past = {-log_likelihood.item():.2f}",
                                    f"ce = {ce.item():.2f}",
                                ]
                            )
                        )

                        optim.zero_grad()
                        (-log_likelihood).backward()
                        optim.step()

                # Predict based on posterior.
                logits = self._forward(episode_batch.slice(t, t + 1), device)
                episode_logits[t] = logits[0].detach()

        return episode_logits
