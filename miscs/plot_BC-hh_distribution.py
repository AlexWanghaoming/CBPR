import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2020_HUMAN_DATA_TRAIN
from overcooked_ai_py.mdp.overcooked_mdp import (OvercookedGridworld,)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import argparse
import torch
from models import BC_MODELS, BCP_MODELS
from bc.bc_hh import BehaviorClone
import gym
from heatmap import plot_heat, plot_combined_heatmaps
from plot_hh_distribution import initialize_distribution, update_distribution, find_empty_space
from cal_prob_distance import cal_tvd_js

def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--layout', default='marshmallow_experiment')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    DEFAULT_DATA_PARAMS = {
        "layouts": [args.layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2019_HUMAN_DATA_TRAIN if args.layout in ['cramped_room',
                                                                   'asymmetric_advantages'] else CLEAN_2020_HUMAN_DATA_TRAIN,
    }
    game_length = 400
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=game_length)
    env = gym.make("Overcooked-v0",
                   base_env=base_env,
                   ego_featurize_fn=base_env.featurize_state_mdp,
                   alt_featurize_fn=base_env.featurize_state_mdp)
    layout_mtx = mdp.terrain_mtx
    n_row, n_col = find_empty_space(layout_mtx)
    # 初始化 四阶段 概率分布矩阵
    ai_prob_mtx = {'stage1':initialize_distribution(n_row, n_col),
                   'stage2': initialize_distribution(n_row, n_col),
                   'stage3': initialize_distribution(n_row, n_col),
                   'stage4': initialize_distribution(n_row, n_col),
                   }
    stage_periodic = 100

    bc_model = torch.load(BC_MODELS[args.layout])
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    num_episodes = len(processed_trajs['ep_returns']) // 2
    total_steps = 0
    for k in range(num_episodes):
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        player1_action_list = processed_trajs['ep_actions'][2*k+1]
        addition = [[0]] * (game_length-len(player1_action_list))
        player1_action_list = player1_action_list + addition
        while not done:
            ai_pos, h_pos = obs['overcooked_state'].player_positions
            ai_pos = tuple(i - 1 for i in ai_pos)  # 位置矫正
            h_pos = tuple(i - 1 for i in h_pos)
            n_stage = episode_steps // stage_periodic + 1
            # 更新位置概率矩阵
            update_distribution(ai_prob_mtx[f'stage{n_stage}'],
                                ai_pos, n_row, n_col, total_steps)
            ai_act = bc_model.choose_action(ai_obs, deterministic=False)
            h_act = player1_action_list[episode_steps][0]
            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            total_steps += 1
            episode_steps += 1
            # env.render(interval=0.05)
        # plot_heat(ai_prob_mtx, n_row, n_col)  # ai轨迹热图
        print(f'Ep {k+1}:',ep_reward)

    plot_combined_heatmaps(ai_prob_mtx, n_row, n_col)
    for idx, (q_s, v_s) in enumerate(ai_prob_mtx.items()):
        for q_e, v_e in list(ai_prob_mtx.items())[idx + 1:]:
            if q_s != q_e:
                print(f'{q_s} -> {q_e}:', cal_tvd_js(np.array(v_s), np.array(v_e)))
                break


