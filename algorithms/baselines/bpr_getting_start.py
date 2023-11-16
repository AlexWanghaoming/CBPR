from typing import Any, Dict, Tuple
from typing_extensions import TypedDict
from ray.tune.registry import register_env
import gym
from gym import spaces
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')
from overcooked_ai_py.mdp.actions import Action
from bpd.envs.overcooked import OvercookedMultiAgent, OvercookedCallbacks
from overcooked_ai_py.mdp.overcooked_mdp import (OvercookedGridworld,)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

# class ApplePickingConfigDict(TypedDict):
#     ring_size: int
#     horizon: int
#
#
# DEFAULT_CONFIG: ApplePickingConfigDict = {"ring_size": 8, "horizon": 100}
#
# class ApplePickingEnv(gym.Env):
#     current_position: int
#     holding_apple: bool
#     timestep: int
#
#     def __init__(self, config: ApplePickingConfigDict = DEFAULT_CONFIG):
#         super().__init__()
#         self.config = {**DEFAULT_CONFIG, **config}
#
#         self.obs_space_size = self.config["ring_size"] + 1
#         self.observation_space = spaces.Box(
#             low=np.zeros(self.obs_space_size, dtype=np.float32),
#             high=np.ones(self.obs_space_size, dtype=np.float32),
#         )
#         self.action_space = spaces.Discrete(2)
#
#     def _get_obs(self) -> np.ndarray:
#         obs = np.zeros(self.obs_space_size)
#         obs[self.current_position] = 1
#         if self.holding_apple:
#             obs[-1] = 1
#         return obs
#
#     def reset(self) -> np.ndarray:
#         self.current_position = 0
#         self.holding_apple = False
#         self.timestep = 0
#         return self._get_obs()
#
#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
#         if action == 0:
#             self.current_position += 1
#             # Wrap around the ring if necessary.
#             if self.current_position >= self.config["ring_size"]:    # 8
#                 self.current_position -= self.config["ring_size"]
#         elif action == 1:
#             self.current_position -= 1
#             # Wrap around the ring if necessary.
#             if self.current_position < 0:
#                 self.current_position += self.config["ring_size"]
#         else:
#             assert False, "Invalid action"
#         reward = 0
#         # The basket is at position 0 in the ring.
#         if self.current_position == 0 and self.holding_apple:
#             # Drop off apple.
#             reward = 1
#             self.holding_apple = False
#         # The tree is halfway around the ring.
#         elif (
#             self.current_position == self.config["ring_size"] // 2
#             and not self.holding_apple
#         ):
#             # Pick apple.
#             self.holding_apple = True
#         self.timestep += 1
#         done = self.timestep >= self.config["horizon"]
#         return self._get_obs(), float(reward), done, {}


# register_env("apple_picking", lambda env_config: ApplePickingEnv(env_config))
mdp = OvercookedGridworld.from_layout_name('cramped_room')
base_env = OvercookedEnv.from_mdp(mdp, horizon=600)
env = gym.make("Overcooked-v0",
               base_env=base_env,
               ego_featurize_fn=base_env.featurize_state_mdp,
               alt_featurize_fn=base_env.featurize_state_mdp)
register_env("Overcooked-v0", lambda env_config: Overcooked(base_env=base_env,
               ego_featurize_fn=base_env.featurize_state_mdp,
               alt_featurize_fn=base_env.featurize_state_mdp))

#
# from typing import List
# from ray.rllib.evaluation import SampleBatch
#
#
# def generate_human_trajectory(action_sequence: List[int]) -> SampleBatch:
#     env = ApplePickingEnv()
#     done = False
#     obs_list = []
#     action_list = []
#     reward_list = []
#     obs = env.reset()
#     timestep = 0
#     while not done:
#         obs_list.append(obs)
#         action = action_sequence[timestep]
#         action_list.append(action)
#         obs, reward, done, info = env.step(action)
#         reward_list.append(reward)
#         timestep += 1
#     return SampleBatch(
#         {
#             SampleBatch.OBS: np.array(obs_list),
#             SampleBatch.ACTIONS: np.array(action_list),
#             SampleBatch.PREV_ACTIONS: np.array([0] + action_list[:-1]),
#             SampleBatch.REWARDS: np.array(reward_list),
#         }
#     )
#
# human_trajectories = [
#     generate_human_trajectory([0] * 100),
#     generate_human_trajectory([1] * 100),
#     generate_human_trajectory(([0] * 8 + [1] * 8) * 7),
# ]

###################################################################################################

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



class ApplePickingDistributionModel(TorchModelV2, nn.Module):
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
            # nn.Linear(
            #     obs_space.shape[0] - latent_size, self.model_config["fcnet_hiddens"][0]
            # ),
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
    "Overcooked-v0",
    ApplePickingDistributionModel,
)


### 计算 BPD
import tqdm
import torch
from bpd.agents.bpd_trainer import BPDTrainer

latent_size = 1000
# Configure the algorithm.
bpd_trainer = BPDTrainer(
    config={
        # "env": "latent_wrapper",
        "env": "overcooked_multi_agent",
        # "env_params": {"horizon": 600},
        "env_config" : {

    # To be passed into OvercookedGridWorld constructor
    "mdp_params": {
        "layout_name": 'cramped_room',
        # "rew_shaping_params": rew_shaping_params,
    },

    # To be passed into OvercookedEnv constructor
    "env_params": {"horizon": 600},

    # To be passed into OvercookedMultiAgent constructor
    "multi_agent_params": {
        "reward_shaping_factor": 1,
        "reward_shaping_horizon": 2e6,
        "use_phi": False,
        "share_dense_reward": False,
        "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
        "extra_rew_shaping": {
            "onion_dispense": 0.0,
            "dish_dispense": 0.0,
        },
        "no_regular_reward": 0.0,
        "action_rewards": [0] * Action.NUM_ACTIONS,
    },
}
,
        "model": {
            "custom_model": "Overcooked-v0",
            "max_seq_len": 10,
            "vf_share_layers": True,
            "custom_model_config": {
                "latent_size": latent_size,
                "discriminate_sequences": True,
            },
            "fcnet_hiddens": [64, 64, 64],
        },
        "temperature": 1,
        "prior_concentration": 1,
        "latent_size": latent_size,
        "framework": "torch",
        "lr": 1e-3,
        "gamma": 0.9,
        "num_sgd_iter": 30,
        "train_batch_size": 100000,
        "sgd_minibatch_size": 8000,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        # "num_gpus":0,
        "create_env_on_driver": True,
        "vf_loss_coeff": 1e-4,
        "kl_coeff":0.2,
        "clip_param":0.05
,
    }
)

for _ in tqdm.trange(50):
    training_result = bpd_trainer.train()
eval_result = bpd_trainer.evaluate()
print("Mean reward = ", eval_result["evaluation"]["episode_reward_mean"])
bpd_trainer.cleanup()


### online prediction
from bpd.latent_prediction import VIActionPredictor

bpd_policy = bpd_trainer.get_policy()

for traj_index, human_trajectory in enumerate(human_trajectories):
    action_logits = VIActionPredictor(bpd_policy).predict_actions(human_trajectory)
    action_distribution = bpd_policy.dist_class(action_logits)
    cross_entropy = -action_distribution.logp(
        human_trajectory[SampleBatch.ACTIONS]
    ).mean()
    print(
        f"Trajectory {traj_index}: Boltzmann policy distribution (using MFVI) cross-entropy = {cross_entropy:.2f}"
    )



