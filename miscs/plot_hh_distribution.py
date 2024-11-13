import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src/')
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2020_HUMAN_DATA_TRAIN
import argparse
import gym
import numpy as np
from typing import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')
from overcooked_ai_py.mdp.overcooked_mdp import (OvercookedGridworld,)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from heatmap import plot_heat, plot_combined_heatmaps
from cal_prob_distance import cal_tvd_js


def find_empty_space(layout):
    # 初始化行和列的最小最大值
    min_row, max_row = len(layout), -1
    min_col, max_col = len(layout[0]), -1

    # 遍历布局以确定空白处的边界
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == ' ':  # 空白处
                # 更新行的最小最大值
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                # 更新列的最小最大值
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    # 计算空白处矩形的行数和列数
    # 如果没有找到空白处，返回(0, 0)
    if min_row <= max_row and min_col <= max_col:
        num_rows = max_row - min_row + 1
        num_cols = max_col - min_col + 1
        return num_rows, num_cols
    else:
        return 0, 0


def initialize_distribution(m, n):
    # 创建并返回初始概率分布矩阵
    initial_probability = 1 / (m * n)
    return [[initial_probability for _ in range(n)] for _ in range(m)]


def update_distribution(matrix, position, m, n, step_count):
    # 更新指定位置的计数
    x, y = position
    # 每个位置的初始概率是1/(n*m)，所以初始计数是1
    # 更新计数后，需要将所有计数除以新的总步数来获得新的概率分布
    # 新的总步数是之前的步数加上1
    new_total_steps = step_count + 1
    # 更新位置的计数
    matrix[y][x] = ((matrix[y][x] * step_count) + 1) / new_total_steps
    # 更新其他位置的概率
    for r in range(m):
        for c in range(n):
            if (c, r) != position:  # 对于非指定位置
                matrix[r][c] = (matrix[r][c] * step_count) / new_total_steps
    return matrix


def merge_two_prob_mtx_dict(mtx1:Dict, mtx2:Dict) -> Dict:
    """
    将mtx2的左边n_cols//2列替换为mtx1的左边n_cols//2列
    """
    n_cols = len(mtx1['stage1'][0])
    for k, v in mtx2.items():
        for row_index in range(len(v)):
            v[row_index][:n_cols//2] = mtx1[k][row_index][:n_cols//2]
    return mtx2


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--layout', type=str, default='random3')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    DEFAULT_DATA_PARAMS = {
        "layouts": [opt.layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2019_HUMAN_DATA_TRAIN if opt.layout in ['cramped_room',
                                                                   'asymmetric_advantages',
                                                                   'random0',
                                                                   'random3'] else CLEAN_2020_HUMAN_DATA_TRAIN,
    }
    game_length = 400
    mdp = OvercookedGridworld.from_layout_name(opt.layout)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=game_length)
    env = gym.make("Overcooked-v0",
                   base_env=base_env)

    layout_mtx = mdp.terrain_mtx
    n_row, n_col = find_empty_space(layout_mtx)

    # 初始化 四阶段 概率分布矩阵
    p0_prob_mtx = {'stage1':initialize_distribution(n_row, n_col),
                   'stage2': initialize_distribution(n_row, n_col),
                   'stage3': initialize_distribution(n_row, n_col),
                   'stage4': initialize_distribution(n_row, n_col),
                   }
    p1_prob_mtx = {'stage1':initialize_distribution(n_row, n_col),
                   'stage2': initialize_distribution(n_row, n_col),
                   'stage3': initialize_distribution(n_row, n_col),
                   'stage4': initialize_distribution(n_row, n_col),
                   }

    stage_periodic = 100
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    num_episodes = len(processed_trajs['ep_returns']) // 2
    total_steps = 0
    for k in range(num_episodes):
        obs = env.reset()
        episode_steps = 0
        done = False
        episode_reward = 0
        player0_action_list = processed_trajs['ep_actions'][2*k]
        player1_action_list = processed_trajs['ep_actions'][2*k+1]
        assert len(player0_action_list) == len(player1_action_list)
        addition = [[0]] * (game_length-len(player1_action_list))
        player0_action_list = player0_action_list + addition
        player1_action_list = player1_action_list + addition
        while not done:
            p0_pos, p1_pos = obs['overcooked_state'].player_positions
            p0_pos = tuple(i-1 for i in p0_pos) # 位置矫正
            p1_pos = tuple(i-1 for i in p1_pos)

            n_stage = episode_steps // stage_periodic + 1

            # 更新位置概率矩阵
            update_distribution(p0_prob_mtx[f'stage{n_stage}'],
                                p0_pos, n_row, n_col, total_steps)
            update_distribution(p1_prob_mtx[f'stage{n_stage}'],
                                p1_pos, n_row, n_col, total_steps)

            p0_act, p1_act = player0_action_list[episode_steps][0], player1_action_list[episode_steps][0]
            obs, sparse_reward, done, info = env.step((p0_act, p1_act))
            r = sparse_reward
            episode_reward += r
            episode_steps += 1
            total_steps += 1
            env.render(interval=0.01)
        # plot_heat(p0_prob_mtx['stage1'], n_row, n_col)
        print(f'Ep {k + 1}:', episode_reward)

    # 对于marshmallow_experiment或者 asymmetric_advantages这种p1 p2 轨迹不重合的layout，合并概率矩阵
    # if opt.layout == 'marshmallow_experiment':
    #     merged_prob_mtx = merge_two_prob_mtx_dict(p1_prob_mtx, p0_prob_mtx)
    #     plot_combined_heatmaps(merged_prob_mtx, n_row, n_col, save_path='H_H_trajs.pdf')
    #
    # ## 计算 JS-散度和总方差变异
    # # plot_combined_heatmaps(p0_prob_mtx, n_row, n_col)
    # for idx, (q_s, v_s) in enumerate(p0_prob_mtx.items()):
    #     for q_e, v_e in list(p0_prob_mtx.items())[idx + 1:]:
    #         if q_s != q_e:
    #             print(f'p0, {q_s} -> {q_e}:', cal_tvd_js(np.array(v_s), np.array(v_e)))
    #             break
    #
    # # plot_combined_heatmaps(p1_prob_mtx, n_row, n_col)
    # for idx, (q_s, v_s) in enumerate(p1_prob_mtx.items()):
    #     for q_e, v_e in list(p1_prob_mtx.items())[idx + 1:]:
    #         if q_s != q_e:
    #             print(f'p1, {q_s} -> {q_e}:', cal_tvd_js(np.array(v_s), np.array(v_e)))
    #             break