# All imports except rllib
from typing import Iterable, cast
from overcooked_ai_py.mdp.actions import Action

import torch
from tqdm import tqdm
from torch.distributions import Dirichlet
from ray.rllib.utils.typing import PolicyID
import numpy as np
import pygame
from scipy.special import logsumexp

from sacred import Experiment

import ray
import os
import skvideo.io
import matplotlib.pyplot as plt
import matplotlib.cm
from ray.rllib.policy.sample_batch import SampleBatch

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from bpd.training_utils import load_trainer
from bpd.agents.bpd_trainer import (
    BPDTrainer,
    project_probabilities,
)
from bpd.envs.overcooked import (
    RlLibAgent,
    get_base_ae,
)
from bpd.models.overcooked_models import (
    OvercookedPPODistributionModel,
    BPDFeaturizeFn,
)
from bpd.agents.bpd_policy import (
    BPDPolicy,
)

ex = Experiment("visualize_overcooked_bpd")


@ex.config
def config():
    checkpoint_path = None  # noqa: F841
    policy_id = "ppo"  # noqa: F841
    episode_len = 400  # noqa: F841
    policy_seed = 0  # noqa: F841
    num_samples = 1000  # noqa: F841
    only_game = False  # noqa: F841
    visualize_discriminator = True  # noqa: F841


def cross_entropy_with_logits(logits, labels):
    return (
        logsumexp(logits, axis=1)
        - np.take_along_axis(logits, labels[:, None], axis=1)[:, 0]
    )


@ex.automain
def main(
    checkpoint_path,
    policy_id: PolicyID,
    episode_len: int,
    policy_seed: int,
    num_samples: int,
    only_game: bool,
    visualize_discriminator: bool,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    # Fix for rendering warning (see https://stackoverflow.com/questions/31847497/pygame-tries-to-use-alsa).
    os.environ["SDL_AUDIODRIVER"] = "dsp"

    trainer = load_trainer(checkpoint_path, run=BPDTrainer)
    policy = cast(BPDPolicy, trainer.get_policy(policy_id))
    model = cast(OvercookedPPODistributionModel, policy.model)

    if "mdp_params" in trainer.config["env_config"]:
        mdp_params = trainer.config["env_config"]["mdp_params"]
    else:
        mdp_params = trainer.config["env_config"]["env_config"]["mdp_params"]

    evaluator = get_base_ae(
        mdp_params,
        {
            "horizon": episode_len,
            "num_mdp": 1,
        },
    )

    latent_vector = np.random.default_rng(seed=policy_seed).normal(
        size=model.latent_size
    )
    featurize_fn = BPDFeaturizeFn(evaluator.env, latent_vector=latent_vector)

    agent0 = RlLibAgent(policy, agent_index=0, featurize_fn=featurize_fn)
    agent1 = RlLibAgent(policy, agent_index=1, featurize_fn=featurize_fn)

    results = evaluator.evaluate_agent_pair(
        AgentPair(agent0, agent1),
        num_games=1,
        # display=False,
        # dir=eval_params["store_dir"],
        # display_phi=eval_params["display_phi"],
    )

    states: Iterable[OvercookedState] = results["ep_states"][0]

    video_fname = os.path.join(
        os.path.dirname(checkpoint_path), f"visualize_{policy_seed}.mp4"
    )
    video_writer = skvideo.io.FFmpegWriter(
        video_fname,
        outputdict={
            "-filter:v": "setpts=15*PTS",
            "-pix_fmt": "yuv420p",
        },
    )
    for state in tqdm(states, desc="Rendering frames"):
        state_frame = pygame.surfarray.array3d(
            StateVisualizer().render_state(state, grid=evaluator.env.mdp.terrain_mtx)
        ).transpose((1, 0, 2))

        if only_game:
            video_writer.writeFrame(state_frame)
        else:
            fig = plt.figure(figsize=(8, 12 if visualize_discriminator else 8))
            state_ax = fig.add_subplot(3 if visualize_discriminator else 2, 1, 1)
            state_ax.imshow(state_frame)
            state_ax.axis("off")

            for agent_index in [0, 1]:
                axs = []

                dist_ax = fig.add_subplot(
                    3 if visualize_discriminator else 2, 2, 3 + agent_index
                )
                axs.append(dist_ax)
                obs = torch.from_numpy(featurize_fn(state)[agent_index]).to(
                    policy.device
                )
                policy_output = model({SampleBatch.OBS: obs[None]})[0].softmax(dim=1)
                input_batch = {
                    SampleBatch.OBS: policy.randomize_latent_in_obs(
                        obs[None].repeat(num_samples, 1, 1, 1),
                        action_space_size=Action.NUM_ACTIONS,
                        all_unique=True,
                    ),
                }
                policy_dist_output = model(input_batch)[0].softmax(dim=1)
                prior_output = Dirichlet(
                    torch.ones_like(policy_dist_output)
                    * policy.config["prior_concentration"]
                ).sample()

                policy_output_projected, vertices = project_probabilities(policy_output)
                policy_dist_output_projected, vertices = project_probabilities(
                    policy_dist_output
                )
                prior_output_projected, _ = project_probabilities(prior_output)

                policy_output_projected = policy_output_projected.detach().cpu()
                policy_dist_output_projected = (
                    policy_dist_output_projected.detach().cpu()
                )
                prior_output_projected = prior_output_projected.detach().cpu()
                vertices = vertices.detach().cpu()

                dist_ax.scatter(
                    policy_dist_output_projected[:, 0],
                    policy_dist_output_projected[:, 1],
                    label="Policy distribution",
                    alpha=0.2,
                )
                dist_ax.scatter(
                    prior_output_projected[:, 0],
                    prior_output_projected[:, 1],
                    label="Prior",
                    alpha=0.2,
                )
                dist_ax.scatter(
                    policy_output_projected[:, 0],
                    policy_output_projected[:, 1],
                    label="Sampled policy",
                    color="k",
                )
                if agent_index == 0:
                    dist_ax.legend(loc="lower left")
                player_color = {0: "#1E6A9E", 1: "#44956B"}[agent_index]
                dist_ax.set_title(f"Player {agent_index + 1}", color=player_color)

                if visualize_discriminator:
                    kl_ax = fig.add_subplot(3, 2, 5 + agent_index)
                    axs.append(kl_ax)
                    discriminator_policy_scores = model.discriminator(
                        {
                            **input_batch,
                            SampleBatch.ACTION_PROB: policy_dist_output,
                        }
                    )[:, 0]
                    discriminator_prior_scores = model.discriminator(
                        {
                            **input_batch,
                            SampleBatch.ACTION_PROB: prior_output,
                        }
                    )[:, 0]
                    kl_ax.scatter(
                        prior_output_projected[:, 0],
                        prior_output_projected[:, 1],
                        c=discriminator_prior_scores.detach().cpu(),
                        cmap=matplotlib.cm.get_cmap("viridis"),
                    )
                    kl_ax.scatter(
                        policy_dist_output_projected[:, 0],
                        policy_dist_output_projected[:, 1],
                        c=discriminator_policy_scores.detach().cpu(),
                        cmap=matplotlib.cm.get_cmap("viridis"),
                    )
                    prior_kl_div = discriminator_policy_scores.mean().item()
                    kl_ax.set_title(f"KL = {prior_kl_div:.2f}")

                for ax in axs:
                    vertices_wrap = torch.cat([vertices, vertices[:1]], dim=0)
                    ax.plot(vertices_wrap[:, 0], vertices_wrap[:, 1], c="k")
                    for action_index, (x, y) in enumerate(vertices.tolist()):
                        ax.annotate(
                            Action.ACTION_TO_CHAR[Action.INDEX_TO_ACTION[action_index]],
                            (x * 0.9, y * 0.9),
                        )
                    ax.axis("off")

            fig.tight_layout()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            video_writer.writeFrame(frame)
    video_writer.close()

    return {"video_fname": video_fname}
