from typing import Dict, List, Optional, Tuple, Union
import gym
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from torch import nn
import numpy as np
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class PickupRingDistributionModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        latent_size: int,
        num_heads: int = 4,
        discriminate_sequences: bool = False,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        assert self.model_config["vf_share_layers"] is True
        assert len(self.model_config["fcnet_hiddens"]) == 3

        self.num_heads = num_heads
        self.latent_size = latent_size
        self.discriminate_sequences = discriminate_sequences

        in_dim = self.obs_space.shape[-1] + self.num_outputs
        self.discriminator_net = self._build_discriminator(in_dim)
        self.detached_discriminator_net = self._build_discriminator(in_dim)

        self.backbone = nn.Sequential(
            nn.Linear(
                obs_space.shape[0] - latent_size, self.model_config["fcnet_hiddens"][0]
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.model_config["fcnet_hiddens"][0],
                self.model_config["fcnet_hiddens"][1],
            ),
            nn.LeakyReLU(),
        )
        self.attention = nn.Linear(
            self.model_config["fcnet_hiddens"][1], self.num_heads * latent_size
        )
        self.head = nn.Sequential(
            nn.Linear(
                self.model_config["fcnet_hiddens"][1] + self.num_heads,
                self.model_config["fcnet_hiddens"][2],
            ),
            nn.LeakyReLU(),
            nn.Linear(self.model_config["fcnet_hiddens"][2], num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                self.model_config["fcnet_hiddens"][1] + self.num_heads,
                self.model_config["fcnet_hiddens"][2],
            ),
            nn.LeakyReLU(),
            nn.Linear(self.model_config["fcnet_hiddens"][2], 1),
        )

    def get_initial_state(self) -> List[np.ndarray]:
        if self.discriminate_sequences:
            return [np.zeros(1)]
        else:
            return super().get_initial_state()

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        self._features = self.backbone(self._last_flat_in[:, : -self.latent_size])
        attention_weights = self.attention(self._features)
        attention_weights = attention_weights.reshape(-1, 4, self.latent_size)
        attention_weights = attention_weights.softmax(-1)

        latent_attention_output = (
            attention_weights * self._last_flat_in[:, None, -self.latent_size :]
        ).sum(2)
        head_input = torch.cat(
            [self._features, latent_attention_output],
            dim=1,
        )

        logits = self.head(head_input)
        self._vf = self.value_head(head_input)[:, 0]

        return logits, [s + 1 for s in state]

    def value_function(self) -> TensorType:
        return self._vf

    def _build_discriminator(
        self,
        in_dim: int,
        out_dim: int = 1,
    ) -> nn.Module:
        if self.discriminate_sequences:
            return TransformerDiscriminator(
                in_dim,
                1,
                self.model_config["fcnet_hiddens"][0],
                len(self.model_config["fcnet_hiddens"]),
            )
        else:
            dims = [in_dim] + self.model_config["fcnet_hiddens"] + [out_dim]
            layers: List[nn.Module] = []
            for dim1, dim2 in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(dim1, dim2))
                layers.append(nn.LeakyReLU())
            layers = layers[:-1]
            return nn.Sequential(*layers)

    def discriminator(
        self, input_dict, seq_lens: Optional[torch.Tensor] = None, detached=False
    ):
        """
        Takes in a dictionary with observations and action probabilities and
        outputs whether it thinks they came from this policy distribution or
        the prior.

        If detached is True, then this will run through a separate, "detached" copy of
        the discriminator which will not propagate gradients to the main network.
        """

        if detached:
            self.detached_discriminator_net.load_state_dict(
                self.discriminator_net.state_dict(keep_vars=False),
            )
            self.detached_discriminator_net.eval()
            discriminator_net = self.detached_discriminator_net
        else:
            discriminator_net = self.discriminator_net

        obs = input_dict[SampleBatch.OBS]
        ac_probs = input_dict[SampleBatch.ACTION_PROB]
        if not detached:
            ac_probs = ac_probs + torch.normal(torch.zeros_like(ac_probs), 0.1)
        net_input = torch.cat([obs, ac_probs], dim=1)
        net_input[
            :, self.obs_space.shape[-1] - self.latent_size : self.obs_space.shape[-1]
        ] = 0

        if self.discriminate_sequences:
            return discriminator_net(net_input, seq_lens)
        else:
            return discriminator_net(net_input)


ModelCatalog.register_custom_model(
    "pickup_ring_distribution_model", PickupRingDistributionModel
)


class TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        size_hidden: int,
        num_layers: int,
        sum_over_seq: bool = True,
    ):
        super().__init__()

        self.sum_over_seq = sum_over_seq

        self.encoder = nn.Linear(in_dim, size_hidden)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=size_hidden,
                nhead=1,
                dim_feedforward=size_hidden,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.head = nn.Linear(size_hidden, out_dim)

    def forward(
        self,
        obs: torch.Tensor,
        seq_lens: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        encoded_obs = self.encoder(obs)

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = encoded_obs.shape[0] // seq_lens.shape[0]
        transformer_inputs = add_time_dimension(
            encoded_obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=False,
        )

        transformer_outputs = self.transformer(transformer_inputs)
        outputs: torch.Tensor = self.head(transformer_outputs)
        if self.sum_over_seq:
            outputs = outputs.sum(dim=1)
        else:
            outputs = outputs.reshape(-1, outputs.size()[-1])
        return outputs
