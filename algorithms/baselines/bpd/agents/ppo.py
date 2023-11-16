from bpd.training_utils import load_policies_from_checkpoint
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer


def after_init(trainer):
    if trainer.config["checkpoint_to_load_policies"] is not None:
        load_policies_from_checkpoint(
            trainer.config["checkpoint_to_load_policies"], trainer
        )


PPOTrainerThatLoadsPolicies = PPOTrainer.with_updates(
    name="PPOTrainerThatLoadsPolicies",
    default_config=Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,
        {"checkpoint_to_load_policies": None},
        _allow_unknown_configs=True,
    ),
    after_init=after_init,
)
