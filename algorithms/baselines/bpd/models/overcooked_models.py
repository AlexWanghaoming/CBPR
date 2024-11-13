import ray.scripts.scripts
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import ModelWeights
from bpd.agents.bpd_policy import ModelWithDiscriminator
from bpd.envs.overcooked import OVERCOOKED_OBS_LAYERS
from typing import List, Optional, OrderedDict, Tuple, Union, cast, Iterator
import warnings
import re
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from gym import spaces
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray import cloudpickle


class OvercookedPPOModel(TorchModelV2, nn.Module):
    # backbone: "SequenceBackbone"
    # action_backbone: "SequenceBackbone"
    # value_backbone: "SequenceBackbone"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        super(OvercookedPPOModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_model_config = {
            **model_config.get("custom_model_config", {}),
            **kwargs,
        }
        self.num_hidden_layers = custom_model_config["num_hidden_layers"]
        self.size_hidden_layers = custom_model_config["size_hidden_layers"]
        self.num_filters = custom_model_config["num_filters"]
        self.num_conv_layers = custom_model_config["num_conv_layers"]
        self.fake_state = custom_model_config.get("fake_state", False)
        self.num_outputs = num_outputs

        if self.num_outputs is None:
            self.num_outputs = self.size_hidden_layers

        self.vf_share_layers: bool = model_config["vf_share_layers"]

        if self.vf_share_layers:
            self.backbone = self._construct_backbone()
        else:
            self.action_backbone = self._construct_backbone()
            self.value_backbone = self._construct_backbone()

        # Linear last layer for action distribution logits
        self.action_head = nn.Linear(
            in_features=self.size_hidden_layers,
            out_features=self.num_outputs,
        )

        # Linear last layer for value function branch of model
        self.value_head = nn.Linear(
            in_features=self.size_hidden_layers,
            out_features=1,
        )

    def _get_obs_space(self) -> spaces.Space:
        return self.obs_space

    def _get_in_channels(self) -> int:
        return cast(int, self._get_obs_space().shape[-1])

    def construct_default_backbone(self, *, scale=1, in_channels=None) -> nn.Sequential:
        # width, height, obs_channels = self._get_obs_space().shape
        print("_get_obs_space:", self._get_obs_space().shape)
        obs_shape = self._get_obs_space().shape[0]
        print("obs_shape:", obs_shape)
        if in_channels is None:
            in_channels = self._get_in_channels()

        # num_filters = int(scale * self.num_filters)
        size_hidden_layers = int(scale * self.size_hidden_layers)

        backbone_layers: List[nn.Module] = []
        # for conv_index in range(self.num_conv_layers):
        #     if conv_index == 0:
        #         padding = (2, 2)
        #     elif conv_index < self.num_conv_layers - 1:
        #         padding = (1, 1)
        #     else:
        #         padding = (0, 0)
        #
        #     backbone_layers.append(
        #         nn.Conv2d(
        #             in_channels=in_channels if conv_index == 0 else num_filters,
        #             out_channels=num_filters,
        #             kernel_size=(5, 5) if conv_index == 0 else (3, 3),
        #             padding=padding,
        #             stride=(1, 1),
        #         )
        #     )
        #     backbone_layers.append(nn.LeakyReLU())
        #
        # backbone_layers.append(nn.Flatten())
        # flattened_conv_size = (width - 2) * (height - 2) * num_filters

        for fc_index in range(self.num_hidden_layers):
            backbone_layers.append(
                nn.Linear(
                    in_features=in_channels if fc_index == 0 else size_hidden_layers,
                    out_features=size_hidden_layers,
                )
            )
            backbone_layers.append(nn.LeakyReLU())

        return nn.Sequential(*backbone_layers)


    def construct_default_vector_backbone(self, *, scale=1, input_dim=None) -> nn.Sequential:
        if input_dim is None:
            input_dim = self._get_input_dim()  # 假设有一个获取输入维度的方法

        num_units = int(scale * self.num_units)  # 使用 num_units 代替 num_filters
        size_hidden_layers = int(scale * self.size_hidden_layers)

        backbone_layers: List[nn.Module] = []
        for fc_index in range(self.num_fc_layers):  # 假设有一个指定全连接层数量的属性
            backbone_layers.append(
                nn.Linear(
                    in_features=input_dim if fc_index == 0 else size_hidden_layers,
                    out_features=size_hidden_layers,
                )
            )
            backbone_layers.append(nn.LeakyReLU())

        # 可选：添加一个输出层
        backbone_layers.append(
            nn.Linear(in_features=size_hidden_layers, out_features=output_dim))  # 假设 output_dim 是输出的维度

        return nn.Sequential(*backbone_layers)

    def _construct_backbone(self) -> nn.Module:
        return self.construct_default_backbone()

    def _get_obs(self, input_dict):
        # obs = (
        #     input_dict[SampleBatch.OBS].permute(0, 3, 1, 2).float()
        # )  # Change to PyTorch NCWH layout
        obs = input_dict[SampleBatch.OBS].float()  # Change to PyTorch NCWH layout
        # if obs.size()[1] > self._get_obs_space().shape[2]:
        #     obs = obs[:, : self._get_obs_space().shape[2]]
        #     warnings.warn("More channels than expected in observation, cropping")
        print("ooooooo", obs.shape)
        return obs

    def get_initial_state(self):
        if self.fake_state:
            return [torch.zeros(1)]
        else:
            return super().get_initial_state()

    def forward(self, input_dict, state, seq_lens):
        self._obs = self._get_obs(input_dict)
        # self._obs = input_dict[SampleBatch.OBS][0]
        # print(input_dict[SampleBatch.OBS][0].shape)
        print("tttttttttttt:", self._obs.shape)

        if self.vf_share_layers:
            self._backbone_out = self.backbone(self._obs)
            logits = self.action_head(self._backbone_out)
        else:
            kk = self.action_backbone(self._obs)
            logits = self.action_head(kk)
        self._logits = logits

        return logits, [s + 1 for s in state]

    def logits(self):
        """
        Get the latest logits output by the model to avoid re-computing them.
        """

        return self._logits

    def value_function(self):
        if self.vf_share_layers:
            return self.value_head(self._backbone_out)[:, 0]
        else:
            return self.value_head(self.value_backbone(self._obs))[:, 0]


ModelCatalog.register_custom_model("overcooked_ppo_model", OvercookedPPOModel)


class OvercookedPPODistributionModel(OvercookedPPOModel, ModelWithDiscriminator):
    """
    Model that randomly samples a policy for each episode by conditioning
    on a multivariate Gaussian vector.
    """
    # backbone: "SequenceBackbone"
    # action_backbone: "SequenceBackbone"
    # value_backbone: "SequenceBackbone"


    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        latent_size: int = 10,
        discriminate_sequences=False,
        split_backbone=False,
        ignore_latents=False,
        discriminator_scale=1,
        use_latent_attention=False,
        pointless_discriminator_latent_input=True,
        **kwargs,
    ):
        self.latent_size = latent_size
        self.split_backbone = split_backbone
        self.discriminate_sequences = discriminate_sequences
        self.ignore_latents = ignore_latents
        self.discriminator_scale = discriminator_scale
        self.use_latent_attention = use_latent_attention
        self.pointless_discriminator_latent_input = pointless_discriminator_latent_input

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        if self.discriminate_sequences:
            self.fake_state = True

        in_channels = self._get_obs_space().shape[-1] + self.num_outputs
        if not (
            self.pointless_discriminator_latent_input
            or isinstance(self.obs_space, spaces.Tuple)
        ):
            in_channels -= self.latent_size
        self.discriminator_net = self._build_discriminator(in_channels)
        self.detached_discriminator_net = self._build_discriminator(in_channels)

        self.initialized = False

    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs)
        self.initialized = True

    def _get_obs_space(self) -> spaces.Space:
        if isinstance(self.obs_space, spaces.Tuple):
            return self.obs_space[0]
        else:
            return self.obs_space

    def _get_obs(self, input_dict):
        raw_obs = input_dict[SampleBatch.OBS]
        if isinstance(raw_obs, (tuple, list)):
            raw_obs, latents = raw_obs
            obs = super()._get_obs({SampleBatch.OBS: raw_obs})
            if self.use_latent_attention:
                return obs, latents
            else:
                obs = torch.cat(
                    [obs,
                     # latents[:, :, None, None].expand(-1, -1, *obs.size()[2:]),
                        latents],
                    dim=1,
                )  # (32, 1096)
        else:
            obs = super()._get_obs(input_dict)
        if self.ignore_latents:
            obs = obs.clone()
            obs[:, -self.latent_size :] = 0
        return obs

    def _construct_backbone(self) -> nn.Module:
        if self.split_backbone:
            return SplitBackbone(self, fc_input_dim=self.latent_size)
        elif self.use_latent_attention:
            return LatentAttentionBackbone(self)
        else:
            return super()._construct_backbone()

    def _build_discriminator(
        self, in_channels: int, out_dim: int = 1, **kwargs
    ) -> nn.Module:
        if self.discriminate_sequences:
            return TransformerDiscriminator(
                self, in_channels, out_dim, scale=self.discriminator_scale, **kwargs
            )
        else:
            return nn.Sequential(
                self.construct_default_backbone(
                    in_channels=in_channels,
                    scale=self.discriminator_scale,
                ),
                nn.Linear(
                    in_features=int(self.size_hidden_layers * self.discriminator_scale),
                    out_features=out_dim,
                ),
            )

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

        raw_obs = input_dict[SampleBatch.OBS]
        obs_includes_latents = True
        if isinstance(raw_obs, (tuple, list)):
            assert not self.pointless_discriminator_latent_input
            raw_obs, _ = raw_obs
            obs_includes_latents = False
        # obs = raw_obs.permute(0, 3, 1, 2).float()  # Change to PyTorch NCWH layout
        obs = raw_obs.float()
        ac_probs = input_dict[SampleBatch.ACTION_PROB]
        print("ac_prob:",ac_probs.shape)
        # ac_probs = ac_probs[:, :, None, None].expand(-1, -1, *obs.size()[2:])

        if self.pointless_discriminator_latent_input:
            net_input = torch.cat([obs, ac_probs], dim=1)
            net_input[
                :,
                self.obs_space.shape[-1] - self.latent_size : self.obs_space.shape[-1],
            ] = 0
        else:
            if obs_includes_latents:
                obs = obs[:, : -self.latent_size]
            net_input = torch.cat([obs, ac_probs], dim=1)   # （32, 102）

        if self.discriminate_sequences:
            return discriminator_net(net_input, seq_lens)
        else:
            print(net_input.shape)
            return discriminator_net(net_input)


ModelCatalog.register_custom_model(
    "overcooked_ppo_distribution_model", OvercookedPPODistributionModel
)


class OvercookedGailModel(OvercookedPPODistributionModel):
    def discriminator(
        self, input_dict, seq_lens: Optional[torch.Tensor] = None, detached=False
    ):
        return super().discriminator(
            {
                SampleBatch.OBS: input_dict[SampleBatch.OBS],
                SampleBatch.ACTION_PROB: F.one_hot(
                    input_dict[SampleBatch.ACTIONS].long(),
                    num_classes=self.action_space.n,
                ),
            },
            seq_lens=seq_lens,
            detached=detached,
        )


ModelCatalog.register_custom_model("overcooked_gail_model", OvercookedGailModel)


class OvercookedSequencePredictionModel(OvercookedPPOModel):
    """
    Used for distillation-prediction, with a transformer or multi-layer LSTM replacing
    the fully-connected backbone.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        use_lstm: bool = False,
        num_heads: int = 2,
        **kwargs,
    ):
        if not model_config.get("vf_share_layers", True):
            warnings.warn(
                "vf_share_layers must be True for "
                "OvercookedSequencePredictionModel, setting it"
            )
        model_config["vf_share_layers"] = True
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.use_lstm = use_lstm
        self.num_heads = num_heads

        assert isinstance(self.backbone, nn.Sequential)
        self.conv_encoder = self.backbone[: self.num_conv_layers * 2 + 2]
        del self.backbone

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.size_hidden_layers,
                hidden_size=self.size_hidden_layers,
                num_layers=self.num_hidden_layers,
                batch_first=True,
            )
        else:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.size_hidden_layers,
                    nhead=self.num_heads,
                    dim_feedforward=self.size_hidden_layers,
                    batch_first=True,
                ),
                num_layers=self.num_hidden_layers,
            )

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS,
            space=self.action_space,
            shift=-1,
        )
        self.view_requirements["prev_obs"] = ViewRequirement(
            SampleBatch.OBS,
            space=self.obs_space,
            shift=-1,
        )

    def _get_in_channels(self) -> int:
        return super()._get_in_channels() * 2 + cast(int, self.action_space.n)

    def _get_obs(self, input_dict):
        obs = super()._get_obs(input_dict)
        prev_obs = super()._get_obs({SampleBatch.OBS: input_dict["prev_obs"]})
        prev_actions = input_dict[SampleBatch.PREV_ACTIONS]
        prev_actions_one_hot = F.one_hot(prev_actions.long(), self.action_space.n)
        prev_actions_one_hot = prev_actions_one_hot[:, :, None, None].expand(
            -1, -1, *obs.size()[2:]
        )
        obs = torch.cat([obs, prev_obs, prev_actions_one_hot], dim=1)
        return obs

    def get_initial_state(self):
        if self.use_lstm:
            weight = self.action_head.weight
            return [
                weight.new(self.num_hidden_layers, self.size_hidden_layers).zero_(),
                weight.new(self.num_hidden_layers, self.size_hidden_layers).zero_(),
            ]
        else:
            return [torch.zeros(1)]

    def forward(self, input_dict, state, seq_lens):
        self._obs = self._get_obs(input_dict)

        encoded_obs = self.conv_encoder(self._obs)

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.tensor(seq_lens).int()
        max_seq_len = encoded_obs.shape[0] // seq_lens.shape[0]
        sequence_inputs = add_time_dimension(
            encoded_obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=False,
        )

        if self.use_lstm:
            h_0, c_0 = (
                state[0].transpose(0, 1).contiguous(),
                state[1].transpose(0, 1).contiguous(),
            )
            sequence_outputs, (h_n, c_n) = self.lstm(sequence_inputs, (h_0, c_0))
            state_out = [h_n.transpose(0, 1), c_n.transpose(0, 1)]
        else:
            sequence_outputs = self.transformer(
                sequence_inputs,
                mask=torch.triu(
                    torch.ones(max_seq_len, max_seq_len, device=self._obs.device),
                    diagonal=1,
                ).bool(),
            )
            state_out = [s + 1 for s in state]

        self._backbone_out = sequence_outputs.flatten(end_dim=1)
        self._logits = self.action_head(self._backbone_out)

        return self._logits, state_out


# Backwards compatibility:
ModelCatalog.register_custom_model(
    "overcooked_transformer_prediction_model", OvercookedSequencePredictionModel
)
ModelCatalog.register_custom_model(
    "overcooked_sequence_prediction_model", OvercookedSequencePredictionModel
)


class OvercookedSequencePolicyModel(OvercookedPPOModel):
    """
    Used for training to cooperate with a policy distribution. A transformer or
    multi-layer LSTM replaces the fully-connected backbone.
    """

    backbone: "SequenceBackbone"
    action_backbone: "SequenceBackbone"
    value_backbone: "SequenceBackbone"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        use_lstm: bool = False,
        num_heads: int = 2,
        memory_inference: int = 400,
        **kwargs,
    ):
        self.use_lstm = use_lstm
        self.num_heads = num_heads
        self.memory_inference = memory_inference

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        self.view_requirements["prev_obs_seq"] = ViewRequirement(
            SampleBatch.OBS,
            space=self.obs_space,
            shift="-{}:-1".format(self.memory_inference),
            used_for_compute_actions=True,
            used_for_training=False,
        )
        self.view_requirements["prev_obs"] = ViewRequirement(
            SampleBatch.OBS,
            space=self.obs_space,
            shift=-1,
            used_for_compute_actions=False,
            used_for_training=True,
        )

    def _get_in_channels(self) -> int:
        return super()._get_in_channels() * 2

    def _construct_backbone(self) -> nn.Module:
        return SequenceBackbone(
            self,
            in_channels=self._get_in_channels(),
            use_lstm=self.use_lstm,
            num_heads=self.num_heads,
        )

    def get_initial_state(self):
        if self.vf_share_layers:
            return self.backbone.get_initial_state()
        else:
            return (
                self.action_backbone.get_initial_state()
                + self.value_backbone.get_initial_state()
            )

    def forward(self, input_dict, state, seq_lens):
        obs = self._get_obs(input_dict)
        if "prev_obs" in input_dict:
            prev_obs = super()._get_obs({SampleBatch.OBS: input_dict["prev_obs"]})
        elif "prev_obs_seq" in input_dict:
            prev_obs_seq = input_dict["prev_obs_seq"].permute(0, 1, 4, 2, 3).float()
            obs_seq = torch.cat([prev_obs_seq, obs[:, None]], dim=1)
            obs = obs_seq[:, 1:].flatten(end_dim=1)
            prev_obs = obs_seq[:, :-1].flatten(end_dim=1)
            seq_lens = torch.ones(
                obs.size()[0], dtype=torch.long, device=obs.device
            ) * (obs_seq.size()[1] - 1)
        else:
            raise ValueError("expected either prev_obs or prev_obs_seq in input_dict")

        backbone_input = torch.cat([obs, prev_obs], dim=1)

        if self.vf_share_layers:
            action_backbone_out, state_out = self.backbone(
                backbone_input, state, seq_lens
            )
            value_backbone_out = action_backbone_out
        else:
            action_state = state[: len(state) // 2]
            value_state = state[len(state) // 2 :]
            action_backbone_out, action_state_out = self.action_backbone(
                backbone_input, action_state, seq_lens
            )
            value_backbone_out, value_state_out = self.value_backbone(
                backbone_input, value_state, seq_lens
            )
            state_out = action_state_out + value_state_out

        if "prev_obs" not in input_dict:
            action_backbone_out_w_time = add_time_dimension(
                action_backbone_out,
                max_seq_len=seq_lens[0],
                framework="torch",
                time_major=False,
            )
            action_backbone_out = action_backbone_out_w_time[:, 0]

        self._logits = self.action_head(action_backbone_out)
        self._vf = self.value_head(value_backbone_out)[:, 0]

        return self._logits, state_out

    def value_function(self):
        return self._vf


ModelCatalog.register_custom_model(
    "overcooked_sequence_policy_model", OvercookedSequencePolicyModel
)


class OvercookedRecurrentStateModel(OvercookedPPOModel):
    """
    Similar to OvercookedPPOModel, but takes additional inputs from the state of
    a pretrained RNN predictive model to incorporate temporal information.
    """

    backbone: "SplitBackbone"
    action_backbone: "SplitBackbone"
    value_backbone: "SplitBackbone"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        recurrent_model_config: dict,
        # Whether inputs to the recurrent model should be from this player or the other
        # player's perspective. If the recurrent model is trained to predict actions
        # from the other player's perspective, then this should be True.
        input_other_player: bool = True,
        **kwargs,
    ):
        recurrent_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            num_outputs,
            recurrent_model_config,
            framework="torch",
        )
        self.initial_state = recurrent_model.get_initial_state()

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.recurrent_model = recurrent_model
        cast(nn.Module, self.recurrent_model).eval()

        self.input_other_player = input_other_player
        self.other_player_layer_permutation = [
            OVERCOOKED_OBS_LAYERS.index(
                re.sub(
                    r"player_(\d)",
                    lambda match: f"player_{1 - int(match.group(1))}",
                    layer_name,
                )
            )
            for layer_name in OVERCOOKED_OBS_LAYERS
        ]

        for key, view_requirement in self.recurrent_model.view_requirements.items():
            if key not in self.view_requirements:
                if key == SampleBatch.PREV_ACTIONS and self.input_other_player:
                    self.view_requirements["infos"] = ViewRequirement(
                        shift=-1,
                        # space=spaces.Dict({"other_action": self.action_space}),
                    )
                else:
                    self.view_requirements[key] = view_requirement

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, nn.Parameter]]:
        # Don't include recurrent model's parameters since we don't want to further
        # optimize them.
        for name, parameter in super().named_parameters(*args, **kwargs):
            if not name.startswith("recurrent_model."):
                yield name, parameter

    def _construct_backbone(self) -> nn.Module:
        size_hidden_state = sum(
            np.prod(state.shape) for state in self.get_initial_state()
        )
        return SplitBackbone(self, fc_input_dim=size_hidden_state)

    def _get_recurrent_model_input_dict(self, input_dict: dict) -> dict:
        if self.input_other_player:
            # Transform input dict to the perspective of the other agent.
            new_input_dict = input_dict.copy()
            new_input_dict[SampleBatch.OBS] = input_dict[SampleBatch.OBS][
                ..., self.other_player_layer_permutation
            ]
            if "prev_obs" in input_dict:
                new_input_dict["prev_obs"] = input_dict["prev_obs"][
                    ..., self.other_player_layer_permutation
                ]
            if SampleBatch.INFOS in input_dict:
                new_input_dict[SampleBatch.PREV_ACTIONS] = torch.tensor(
                    [
                        info["other_action"] if isinstance(info, dict) else 0
                        for info in input_dict[SampleBatch.INFOS]
                    ],
                    device=input_dict[SampleBatch.OBS].device,
                )
            return new_input_dict
        else:
            return input_dict

    def get_initial_state(self):
        return self.initial_state

    def forward(self, input_dict, state, seq_lens):
        obs = self._get_obs(input_dict)

        if seq_lens is not None and any(seq_len > 1 for seq_len in seq_lens):
            # This should only happen during loss function initialization.
            state_out = [
                state[None].repeat_interleave(obs.size()[0], dim=0).to(obs.device)
                for state in self.get_initial_state()
            ]
        else:
            _, state_out = self.recurrent_model(
                self._get_recurrent_model_input_dict(input_dict),
                state,
                seq_lens,
            )

        flat_state_out = torch.cat(state_out, dim=-1).flatten(start_dim=1)

        if self.vf_share_layers:
            action_backbone_out = self.backbone(obs, flat_state_out)
            value_backbone_out = action_backbone_out
        else:
            action_backbone_out = self.action_backbone(obs, flat_state_out)
            value_backbone_out = self.value_backbone(obs, flat_state_out)

        self._logits = self.action_head(action_backbone_out)
        self._vf = self.value_head(value_backbone_out)[:, 0]

        return self._logits, state_out

    def value_function(self):
        return self._vf

    def load_recurrent_model_from_checkpoint(
        self, checkpoint_fname: str, policy_id: str
    ) -> None:
        with open(checkpoint_fname, "rb") as checkpoint_file:
            checkpoint_data = cloudpickle.load(checkpoint_file)
        model_weights: ModelWeights = cloudpickle.loads(checkpoint_data["worker"])[
            "state"
        ][policy_id]

        if "_optimizer_variables" in model_weights:
            del model_weights["_optimizer_variables"]
        model_weights_tensor: OrderedDict[str, torch.Tensor] = convert_to_torch_tensor(
            model_weights, device=self.action_head.weight.device  # type: ignore
        )
        cast(nn.Module, self.recurrent_model).load_state_dict(model_weights_tensor)


ModelCatalog.register_custom_model(
    "overcooked_recurrent_state_model", OvercookedRecurrentStateModel
)


class RandomPolicyModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fake_state=False,
        **kwargs,
    ):
        super(RandomPolicyModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self.fake_state = fake_state

        # Avoid errors in optimizer.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def get_initial_state(self):
        if self.fake_state:
            return [torch.zeros(1)]
        else:
            return super().get_initial_state()

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict[SampleBatch.OBS]
        state_out = [s + 1 for s in state]
        if isinstance(obs, (list, tuple)):
            logits = obs[1][:, -self.num_outputs :]
        else:
            logits = obs.flatten(start_dim=1, end_dim=-2)[:, 0, -self.num_outputs :]
        self._vf = torch.zeros_like(logits[:, 0])
        return logits, state_out

    def value_function(self):
        return self._vf


ModelCatalog.register_custom_model("random_model", RandomPolicyModel)


class SplitBackbone(nn.Module):
    """
    Similar to the backbone implemented in OvercookedPPODistributionModel, but splits
    part of the input into a separate fully connected network and concatenates with the
    output of the convolutional network for the observation.
    """

    def __init__(
        self,
        model: OvercookedPPOModel,
        fc_input_dim: Optional[int] = None,
        num_fc_encoder_layers: Optional[int] = None,
    ):
        super().__init__()

        if fc_input_dim is None:
            assert isinstance(model, OvercookedPPODistributionModel)
            fc_input_dim = model.latent_size
        if num_fc_encoder_layers is None:
            num_fc_encoder_layers = model.num_conv_layers
        self.fc_input_dim = fc_input_dim

        sequential_backbone = model.construct_default_backbone(
            in_channels=model._get_in_channels()
        )
        num_encoder_layers = model.num_conv_layers * 2 + 1
        self.obs_encoder = sequential_backbone[:num_encoder_layers]

        fc_encoder_layers: List[nn.Module] = []
        for fc_index in range(num_fc_encoder_layers):
            fc_encoder_layers.append(
                nn.Linear(
                    in_features=self.fc_input_dim
                    if fc_index == 0
                    else model.size_hidden_layers,
                    out_features=model.size_hidden_layers,
                )
            )
            fc_encoder_layers.append(nn.LeakyReLU())
        self.fc_encoder = nn.Sequential(*fc_encoder_layers)

        self.fc_backbone = sequential_backbone[num_encoder_layers:]
        first_fc_layer: nn.Linear = self.fc_backbone[0]
        self.fc_backbone[0] = nn.Linear(
            in_features=first_fc_layer.in_features + model.size_hidden_layers,
            out_features=model.size_hidden_layers,
        )

    def forward(
        self, obs: torch.Tensor, fc_inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if fc_inputs is None:
            fc_inputs = obs[:, -self.fc_input_dim :, 0, 0]
            obs = obs[:, : -self.fc_input_dim]

        encoded_obs = self.obs_encoder(obs)
        encoded_fc = self.fc_encoder(fc_inputs)
        return cast(
            torch.Tensor, self.fc_backbone(torch.cat([encoded_obs, encoded_fc], dim=1))
        )


class LatentAttentionBackbone(nn.Module):
    """
    Calculates an "attention" matrix over the latent variables and then uses this to
    form several convex combinations of them and input those to the fully connected
    part.
    """

    def __init__(
        self,
        model: OvercookedPPODistributionModel,
        num_heads: int = 10,
    ):
        self.num_heads = num_heads
        self.latent_size = model.latent_size
        super().__init__()

        in_channels = model._get_in_channels()
        if not isinstance(model.obs_space, spaces.Tuple):
            in_channels -= self.latent_size
        sequential_backbone = model.construct_default_backbone(in_channels=in_channels)
        num_encoder_layers = model.num_conv_layers * 2 + 3
        self.obs_encoder = sequential_backbone[:num_encoder_layers]

        self.attention_head = nn.Linear(
            in_features=model.size_hidden_layers,
            out_features=num_heads * self.latent_size,
        )

        self.fc_backbone = sequential_backbone[num_encoder_layers:]
        self.fc_backbone[0] = nn.Linear(
            in_features=model.size_hidden_layers + num_heads,
            out_features=model.size_hidden_layers,
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is None:
            if isinstance(obs, (tuple, list)):
                obs, latents = obs
            else:
                latents = obs[:, -self.latent_size :, 0, 0]
                obs = obs[:, : -self.latent_size]

        encoded_obs = self.obs_encoder(obs)

        attention_weights = self.attention_head(encoded_obs)
        attention_weights = attention_weights.reshape(
            -1, self.num_heads, self.latent_size
        )
        attention_weights = attention_weights.softmax(-1)

        latent_attention_output = (attention_weights * latents[:, None, :]).sum(2)
        return cast(
            torch.Tensor,
            self.fc_backbone(torch.cat([encoded_obs, latent_attention_output], dim=1)),
        )


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        model: OvercookedPPOModel,
        in_channels: int,
        use_lstm: bool = False,
        num_heads: int = 10,
    ):
        super().__init__()

        self.use_lstm = use_lstm
        self.num_heads = num_heads

        backbone = model.construct_default_backbone(in_channels=in_channels)
        assert isinstance(backbone, nn.Sequential)
        self.conv_encoder = backbone[: model.num_conv_layers * 2 + 2]

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=model.size_hidden_layers,
                hidden_size=model.size_hidden_layers,
                num_layers=model.num_hidden_layers,
                batch_first=True,
            )
        else:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=model.size_hidden_layers,
                    nhead=self.num_heads,
                    dim_feedforward=model.size_hidden_layers,
                    batch_first=True,
                ),
                num_layers=model.num_hidden_layers,
            )

        if self.use_lstm:
            weight = self.conv_encoder[0].weight
            self._initial_state = [
                weight.new(model.num_hidden_layers, model.size_hidden_layers).zero_(),
                weight.new(model.num_hidden_layers, model.size_hidden_layers).zero_(),
            ]
        else:
            self._initial_state = [torch.zeros(1)]

    def get_initial_state(self):
        return self._initial_state

    def forward(self, obs, state, seq_lens):
        encoded_obs = self.conv_encoder(obs)

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.tensor(seq_lens).int()
        max_seq_len = encoded_obs.shape[0] // seq_lens.shape[0]
        sequence_inputs = add_time_dimension(
            encoded_obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=False,
        )

        if self.use_lstm:
            h_0, c_0 = (
                state[0].transpose(0, 1).contiguous(),
                state[1].transpose(0, 1).contiguous(),
            )
            sequence_outputs, (h_n, c_n) = self.lstm(sequence_inputs, (h_0, c_0))
            state_out = [h_n.transpose(0, 1), c_n.transpose(0, 1)]
        else:
            sequence_outputs = self.transformer(
                sequence_inputs,
                mask=torch.triu(
                    torch.ones(max_seq_len, max_seq_len, device=obs.device),
                    diagonal=1,
                ).bool(),
            )
            state_out = [s + 1 for s in state]

        return sequence_outputs.flatten(end_dim=1), state_out


class TransformerDiscriminator(nn.Module):
    """
    Similar to the backbone implemented in OvercookedPPODistributionModel, but splits
    part of the input into a separate fully connected network and concatenates with the
    output of the convolutional network for the observation.
    """

    def __init__(
        self,
        model: OvercookedPPODistributionModel,
        in_channels: int,
        out_dim: int,
        sum_over_seq: bool = True,
        scale: float = 1,
    ):
        super().__init__()

        self.sum_over_seq = sum_over_seq

        backbone = model.construct_default_backbone(
            in_channels=in_channels,
            scale=scale,
        )
        print('hhhhhhhh', model.num_conv_layers)
        self.conv_encoder = backbone[: model.num_conv_layers * 2 + 2]

        size_hidden_layers = int(model.size_hidden_layers * scale)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=size_hidden_layers,
                nhead=1,
                dim_feedforward=size_hidden_layers,
                batch_first=True,
            ),
            num_layers=model.num_hidden_layers,
        )

        self.head = nn.Linear(size_hidden_layers, out_dim)

    def forward(
        self,
        obs: torch.Tensor,
        seq_lens: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        encoded_obs = self.conv_encoder(obs)

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


def append_latent_vector(obs, latent_vector: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        np.concatenate(
            [
                obs,
                np.resize(
                    latent_vector[:, np.newaxis, np.newaxis],
                    obs.shape[:-1] + latent_vector.shape,
                ),
            ],
            axis=2,
        ),
    )


BPDObs = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


class BPDFeaturizeFn(object):
    latent_vector: np.ndarray

    def __init__(
        self,
        env: OvercookedEnv,
        *,
        latent_vector: Optional[np.ndarray] = None,
        latent_size: Optional[int] = None,
        use_tuple: bool = False,
    ):
        self.env = env
        if latent_vector is not None:
            self.latent_vector = latent_vector
            self.sample_per_episode = False
            self.latent_size = latent_vector.shape[0]
        elif latent_size is not None:
            self.sample_per_episode = True
            self.latent_size = latent_size
        else:
            raise ValueError("Must specify either latent_vector or latent_size")
        self.use_tuple = use_tuple

    def __call__(self, state: OvercookedState) -> Tuple[BPDObs, BPDObs]:
        if state.timestep == 0 and self.sample_per_episode:
            self.latent_vector = np.random.default_rng().normal(size=self.latent_size)
        obs0, obs1 = self.env.lossless_state_encoding_mdp(state)
        if self.use_tuple:
            return (obs0, self.latent_vector), (obs1, self.latent_vector)
        else:
            return (
                append_latent_vector(obs0, self.latent_vector),
                append_latent_vector(obs1, self.latent_vector),
            )
