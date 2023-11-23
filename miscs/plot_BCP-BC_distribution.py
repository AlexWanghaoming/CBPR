import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')
from overcooked_ai_py.mdp.overcooked_mdp import (OvercookedGridworld,)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import argparse
import torch
from models import BC_MODELS, BCP_MODELS
from bc.bc_hh import BehaviorClone
import gym
from heatmap import plot_heat, plot_combined_heatmaps
from plot_hh_distribution import initialize_distribution, update_distribution, find_empty_space, merge_two_prob_mtx_dict
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
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=26)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    game_length = 400
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=game_length)
    env = gym.make("Overcooked-v0",
                   base_env=base_env)
    layout_mtx = mdp.terrain_mtx
    n_row, n_col = find_empty_space(layout_mtx)

    # 初始化 四阶段 概率分布矩阵
    ai_prob_mtx = {'stage1': initialize_distribution(n_row, n_col),
                   'stage2': initialize_distribution(n_row, n_col),
                   'stage3': initialize_distribution(n_row, n_col),
                   'stage4': initialize_distribution(n_row, n_col),
                   }
    h_prob_mtx = {'stage1': initialize_distribution(n_row, n_col),
                   'stage2': initialize_distribution(n_row, n_col),
                   'stage3': initialize_distribution(n_row, n_col),
                   'stage4': initialize_distribution(n_row, n_col),
                   }
    stage_periodic = 100
    # ai_prob_mtx = initialize_distribution(n_row, n_col)
    # h_prob_mtx = initialize_distribution(n_row, n_col)
    bc_model = torch.load(BC_MODELS[args.layout])
    ai_agent = torch.load(BCP_MODELS[args.layout])
    total_steps = 0
    for k in range(args.num_episodes):
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        while not done:
            ai_pos, h_pos = obs['overcooked_state'].player_positions
            ai_pos = tuple(i - 1 for i in ai_pos)  # 位置矫正
            h_pos = tuple(i - 1 for i in h_pos)
            
            n_stage = episode_steps//stage_periodic + 1
            # 更新位置概率矩阵
            update_distribution(ai_prob_mtx[f'stage{n_stage}'],
                                ai_pos, n_row, n_col, total_steps)
            update_distribution(h_prob_mtx[f'stage{n_stage}'],
                                h_pos, n_row, n_col, total_steps)
            ai_act = evaluate(ai_agent, ai_obs)
            h_act = bc_model.choose_action(h_obs, deterministic=False)
            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            episode_steps += 1
            total_steps += 1
            # env.render(interval=0.05)
        # plot_heat(ai_prob_mtx['stage1'], n_row, n_col)
        # plot_heat(h_prob_mtx['stage1'], n_row, n_col)
        print(f'Ep {k + 1}:', ep_reward)

    # 对于marshmallow_experiment或者 asymmetric_advantages这种p1 p2 轨迹不重合的layout，合并概率矩阵
    if args.layout == 'marshmallow_experiment':
        merged_prob_mtx = merge_two_prob_mtx_dict(h_prob_mtx, ai_prob_mtx)
        plot_combined_heatmaps(merged_prob_mtx, n_row, n_col, save_path='BCP_BC_trajs.pdf')

    # plot_combined_heatmaps(ai_prob_mtx, n_row, n_col)
    for idx, (q_s, v_s) in enumerate(ai_prob_mtx.items()):
        for q_e, v_e in list(ai_prob_mtx.items())[idx + 1:]:
            if q_s != q_e:
                print(f'BCP, {q_s} -> {q_e}:', cal_tvd_js(np.array(v_s), np.array(v_e)))
                break
                
    # plot_combined_heatmaps(h_prob_mtx, n_row, n_col)
    for idx, (q_s, v_s) in enumerate(h_prob_mtx.items()):
        for q_e, v_e in list(h_prob_mtx.items())[idx + 1:]:
            if q_s != q_e:
                print(f'BC, {q_s} -> {q_e}:', cal_tvd_js(np.array(v_s), np.array(v_e)))
                break

