import os
from typing import Optional, Type, Union, cast
import gym
from datetime import datetime
import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.policy import Policy
from ray.rllib.rollout import rollout
from ray.rllib.utils.typing import PolicyID
from sacred import Experiment
from sacred import SETTINGS

from ..training_utils import load_trainer
from .evaluate_overcooked import RunStr, trainer_classes

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("rollout")


@ex.config
def sacred_config():
    run = "PPO"  # noqa: F841
    checkpoint = ""
    episodes = 100  # noqa: F841
    experiment_name = None

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_name is not None:
        rollouts_dir = f"rollouts_{experiment_name}_{time_str}"
    else:
        rollouts_dir = f"rollouts_{time_str}"
    out_dir = os.path.join(os.path.dirname(checkpoint), rollouts_dir)

    num_workers = 4
    output_max_file_size = 64 * 1024 * 1024
    config_updates = {  # noqa: F841
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_num_episodes": max(num_workers, 1),
        "output_max_file_size": output_max_file_size,
        "evaluation_config": {},
        "custom_eval_function": None,
        "output_compress_columns": ["prev_obs", "obs", "new_obs", "infos"],
        "output": out_dir,
    }


@ex.automain
def main(
    run: Union[str, Type[Trainer]],
    config_updates: dict,
    checkpoint: str,
    experiment_name: Optional[str],
    episodes: int,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    if isinstance(run, str) and run in trainer_classes:
        run = trainer_classes[cast(RunStr, run)]

    trainer = load_trainer(checkpoint, run, config_overrides=config_updates)
    evaluation_workers = trainer.evaluation_workers
    assert evaluation_workers is not None

    # Remove extraneous view requirements like state_in_* and state_out_*.
    def remove_extra_view_requirements(policy: Policy, policy_id: PolicyID):
        for key in list(policy.view_requirements):
            if key.startswith("state_in_") or key.startswith("state_out_"):
                del policy.view_requirements[key]

    evaluation_workers.foreach_policy(remove_extra_view_requirements)

    gym.logger.set_level(gym.logger.INFO)

    rollout(
        trainer,
        None,
        num_steps=0,
        num_episodes=episodes,
    )
    trainer.stop()
