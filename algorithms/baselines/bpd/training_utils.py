import os
from pickle import Unpickler
import shutil
import numpy as np
import ray.cloudpickle as cloudpickle
from datetime import datetime
from typing import Any, Callable, Dict, Type, Union, cast
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID, TrainerConfigDict
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls
from ray.rllib.agents.trainer import COMMON_CONFIG


def build_logger_creator(log_dir: str, experiment_name: str):
    experiment_dir = os.path.join(
        log_dir,
        experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in
        <log_dir>/<experiment_name>_<timestamp>
        """

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        return UnifiedLogger(config, experiment_dir)

    return custom_logger_creator


class OldCheckpointUnpickler(Unpickler):
    """
    A bit of a hacky workaround for loading old config files that might have pickled
    representations of functions that no longer exist.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("maxent_policy_rl."):
            module = "bpd." + module[len("maxent_policy_rl.") :]
        if (module, name) in [
            ("bpd.agents.bc", "setup_mixins"),
            ("bpd.agents.bc", "bc_loss"),
            ("bpd.agents.bc", "stats"),
        ]:
            return lambda *args, **kwargs: None
        elif module == "ray.rllib.utils.torch_ops":
            module = "ray.rllib.utils.torch_utils"
        return super().find_class(module, name)


def load_trainer_config(checkpoint_path: str) -> TrainerConfigDict:
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory!"
        )
    # Load the config from pickled.
    with open(config_path, "rb") as f:
        config: TrainerConfigDict = OldCheckpointUnpickler(f).load()

    # This is a fix because we had to change where distillation_mapping_fn is put in
    # the config dict.
    if "distillation_mapping_fn" in config.get("multiagent", {}):
        config["distillation_mapping_fn"] = config["multiagent"].pop(
            "distillation_mapping_fn"
        )

    # Another fix for newer versions of RLlib.
    if "disable_env_checking" in COMMON_CONFIG:
        config.setdefault("disable_env_checking", True)

    # Fix to remove some old keys in the config dict.
    for old_key in [
        "min_temperature",
        "max_temperature",
        "target_reward",
        "input_random_policy",
        "use_denoiser",
        "denoiser_sigma",
    ]:
        if old_key in config:
            del config[old_key]

    return config


def convert_checkpoint(checkpoint_fname: str) -> str:
    """
    RLlib changed the way policies are saved at some point since running most of the
    experiments (https://github.com/ray-project/ray/pull/16354) so this will
    automatically convert old-style checkpoints to the new format, and return a new
    file with the converted checkpoint if necessary.
    """

    with open(checkpoint_fname, "rb") as checkpoint_file:
        checkpoint_data = cloudpickle.load(checkpoint_file)
    worker_data = cloudpickle.loads(checkpoint_data["worker"])
    policy_states: Dict[str, Any] = worker_data["state"]
    if all(("weights" in policy_state) for policy_state in policy_states.values()):
        return checkpoint_fname
    else:
        converted_fname = checkpoint_fname + "-converted"
        if not os.path.exists(converted_fname):
            policy_state: dict
            for policy_id, policy_state in policy_states.items():
                policy_state["global_timestep"] = checkpoint_data["train_exec_impl"][
                    "counters"
                ].get("num_steps_trained", 0)
                policy_state["weights"] = {}
                keys_to_remove = []
                for state_key, state_value in policy_state.items():
                    # Assume that anything that's a NumPy array is a model weight.
                    if isinstance(state_value, np.ndarray):
                        policy_state["weights"][state_key] = policy_state[state_key]
                        keys_to_remove.append(state_key)
                for state_key in keys_to_remove:
                    del policy_state[state_key]
            checkpoint_data["worker"] = cloudpickle.dumps(worker_data)
            with open(converted_fname, "wb") as converted_file:
                cloudpickle.dump(checkpoint_data, converted_file)
            shutil.copy(
                checkpoint_fname + ".tune_metadata", converted_fname + ".tune_metadata"
            )
        return converted_fname


def load_trainer(
    checkpoint_path: str,
    run: Union[str, Type[Trainer]],
    config_overrides: Dict[str, Any] = {},
) -> Trainer:
    config = load_trainer_config(checkpoint_path)

    config["num_workers"] = 0
    config.update(config_overrides)

    # Create the Trainer from config.
    if isinstance(run, str):
        cls = cast(Type[Trainer], get_trainable_cls(run))
    else:
        cls = run
    trainer: Trainer = cls(config=config)

    # Load state from checkpoint.
    trainer.restore(convert_checkpoint(checkpoint_path))

    return trainer


def load_policies_from_checkpoint(
    checkpoint_fname: str,
    trainer: Trainer,
    policy_map: Callable[[PolicyID], PolicyID] = lambda policy_id: policy_id,
):
    """
    Load policy model weights from a checkpoint and copy them into the given
    trainer.
    """

    with open(convert_checkpoint(checkpoint_fname), "rb") as checkpoint_file:
        checkpoint_data = cloudpickle.load(checkpoint_file)
    policy_states: Dict[str, Any] = cloudpickle.loads(checkpoint_data["worker"])[
        "state"
    ]

    policy_weights = {
        policy_map(policy_id): policy_state["weights"]
        for policy_id, policy_state in policy_states.items()
    }

    def copy_policy_weights(policy: Policy, policy_id: PolicyID):
        if policy_id in policy_weights:
            policy.set_weights(policy_weights[policy_id])

    workers: WorkerSet = cast(Any, trainer).workers
    workers.foreach_policy(copy_policy_weights)
