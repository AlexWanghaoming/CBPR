import ray
from packaging import version
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import SelectExperiences


def get_select_experiences(workers: WorkerSet) -> SelectExperiences:
    if version.parse(ray.__version__) < version.parse("1.12.0"):
        return SelectExperiences(workers.trainable_policies())
    else:
        return SelectExperiences(local_worker=workers.local_worker())  # type: ignore
