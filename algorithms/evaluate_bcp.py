import numpy as np
import re, pickle
from typing import List, Tuple, Dict, Optional, Type, Union
import argparse
import torch
import torch.nn as nn
import os, sys
from models import META_TASK_MODELS
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from bc.bc_hh import BehaviorClone
from agents.ppo_discrete import PPO_discrete
from utils import seed_everything, init_env
from rl_plotter.logger import Logger
import random
import wandb
from datetime import datetime

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
    parser.add_argument('--layout', default='asymmetric_advantages')

    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n

    bc_models = []
    for i in range(4):
        bc_models.append(torch.load(META_TASK_MODELS[args.layout][i]))  # wanghm GAIL载入模型时不释放内存，需要提前存到列表

    ai_agent = torch.load(f'../models/bcp/bcp_{args.layout}-seed42.pth')
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='bpr_NN',
               name=f'BCP_{args.layout}_{args.mode}-{args.switch_human_freq}_{formatted_now}',
               config=vars(args))
    # seeds = [0, 1, 42, 2022, 2023]
    seeds = [1]
    for seed in seeds:
        if args.mode == 'inter':
            random.seed(55)
            N = args.num_episodes // args.switch_human_freq
            policy_id_list = [random.randint(1, 4) for _ in range(N)]  # 固定测试的策略顺序
            policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
        else:
            random.seed(55)
            N = args.num_episodes * (600 // args.switch_human_freq)
            policy_id_list = [random.randint(1, 4) for _ in range(N)]

        seed_everything(seed)
        # logger = Logger(log_dir=f'./logs/bcp/{args.layout}',
        #                 exp_name=f'BCP-switch-{args.mode}-{args.switch_human_freq}',
        #                 env_name='')

        # 初始化人类模型和策略
        policy_idx = 1
        bc_model = bc_models[policy_idx - 1]
        for k in range(args.num_episodes):
            if args.mode == 'inter':
                # 每switch_human_freq 改变一次策略
                policy_idx = policy_id_list.pop()
                print("人类改变至策略: ", policy_idx) #log
                bc_model = bc_models[policy_idx - 1]
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            episode_steps = 0
            while not done:
                episode_steps += 1
                if args.mode == "intra":
                    # 轮内切换人的策略
                    if episode_steps % args.switch_human_freq == 0:
                        policy_idx = policy_id_list.pop()
                        print(f"人类改变至策略:  {policy_idx}")
                        bc_model = bc_models[policy_idx - 1]

                ai_act = evaluate(ai_agent, ai_obs)
                h_act = bc_model.choose_action(h_obs)
                obs, sparse_reward, done, info = env.step((ai_act, h_act))
                ai_obs, h_obs = obs['both_agent_obs']
                ep_reward += sparse_reward

            print(f'Ep {k+1}:',ep_reward)
            wandb.log({'episode': k+1, 'ep_reward': ep_reward})
            # logger.update(score=[ep_reward], total_steps=k + 1)


