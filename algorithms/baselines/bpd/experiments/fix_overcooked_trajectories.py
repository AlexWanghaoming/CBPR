# All imports except rllib
from bpd.envs.overcooked import (
    HumanData,
    OvercookedAction,
    OvercookedMultiAgent,
)
from typing import List, Tuple

import pickle

from sacred import Experiment

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

ex = Experiment("fix_overcooked_trajectories")


@ex.config
def config():
    in_fname = None  # noqa: F841
    out_fname = None  # noqa: F841


@ex.automain
def main(
    in_fname: str,
    out_fname: str,
    _log,
):
    with open(in_fname, "rb") as human_data_file:
        human_data: HumanData = pickle.load(human_data_file)

    fixed_human_data: HumanData = {}

    for layout_name, layout_trajectories in human_data.items():
        try:
            env = OvercookedMultiAgent.from_config(
                {
                    "mdp_params": {
                        "layout_name": layout_name,
                    },
                    "env_params": {"horizon": 1e10},
                    "multi_agent_params": {},
                }
            )
        except FileNotFoundError:
            # Tried to load a nonexistent layout.
            continue
        base_env = env.base_env

        fixed_layout_trajectories: List[List[Tuple[dict, List[OvercookedAction]]]] = []
        for traj_index, trajectory in enumerate(layout_trajectories):
            fixed_trajectory = trajectory[:1]
            _log.info(f"Fixing trajectory {traj_index} for {layout_name}")

            base_env.state = OvercookedState.from_dict(trajectory[0][0])

            for t, ((_, actions), (new_state_dict, new_actions)) in enumerate(
                zip(trajectory[:-1], trajectory[1:])
            ):
                new_state = OvercookedState.from_dict(new_state_dict)
                new_state.timestep = t + 1
                next_state: OvercookedState
                next_state, reward, done, info = base_env.step(actions)
                if new_state != next_state:
                    print(
                        "States differed (expected vs actual): "
                        f"{base_env.display_states(new_state, next_state)}\n\n"
                        f"expected dict: \t{new_state.to_dict()}\n"
                        f"actual dict: \t{next_state.to_dict()}\n"
                        f"actions: \t{actions}"
                    )
                fixed_trajectory.append((next_state.to_dict(), new_actions))
            fixed_layout_trajectories.append(fixed_trajectory)
        fixed_human_data[layout_name] = fixed_layout_trajectories

    with open(out_fname, "wb") as out_file:
        pickle.dump(fixed_human_data, out_file)
