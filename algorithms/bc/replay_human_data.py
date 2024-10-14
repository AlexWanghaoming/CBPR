import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src/')
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2020_HUMAN_DATA_TRAIN
import argparse
from My_utils import init_env
from typing import Dict, Tuple, List


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--layout', type=str, default='soup_coordination')
    # parser.add_argument('--layout', type=str, default='asymmetric_advantages')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    DEFAULT_DATA_PARAMS = {
        "layouts": [opt.layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2019_HUMAN_DATA_TRAIN if opt.layout in ['cramped_room', 'asymmetric_advantages'] else CLEAN_2020_HUMAN_DATA_TRAIN,
    }
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    game_length = 400
    env = init_env(layout=opt.layout, horizon=game_length)
    num_episodes = len(processed_trajs['ep_returns']) // 2
    for k in range(num_episodes):
        obs = env.reset()
        episode_steps = 0
        done = False
        episode_reward = 0
        agent_env_steps = 600 * k
        player0_action_list = processed_trajs['ep_actions'][2*k]
        player1_action_list = processed_trajs['ep_actions'][2*k+1]

        assert len(player0_action_list) == len(player1_action_list)
        addition = [[0]] * (game_length-len(player1_action_list))

        player0_action_list = player0_action_list + addition
        player1_action_list = player1_action_list + addition
        while not done:
            obs_, sparse_reward, done, info = env.step((player0_action_list[episode_steps][0],
                                                        player1_action_list[episode_steps][0]))
            # shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward
            episode_reward += r
            episode_steps += 1
            env.render(interval=0.08)
        print(f'Ep {k+1}:', episode_reward)