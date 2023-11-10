import numpy as np
import re, pickle
from typing import List, Tuple, Dict, Optional, Type, Union
import argparse
import torch
import torch.nn as nn
import os, sys
from models import MTP_MODELS, META_TASK_MODELS
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from bc.bc_hh import BehaviorClone
# from agents.ppo_discrete import PPO_discrete
from utils import seed_everything, init_env


def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='marshmallow_experiment', help='layout name')
    parser.add_argument('--num_episodes', type=int, default=100, help='total episodes')
    # parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    # parser.add_argument("--switch_human_freq", type=int, default=50, help="Frequency of switching human policy")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n

    # ai_agent = torch.load(f'../models/bcp/bcp_{args.layout}-seed42.pth')
    # ai_agent = torch.load(f'../models/mtp/mtp_{args.layout}-vaeCluster1-seed42.pth')

    ai_agent = torch.load(META_TASK_MODELS[args.layout][2])
    bc_model = torch.load(META_TASK_MODELS[args.layout][1])

    # logger = Logger(log_dir=f'./logs/bcp/{LAYOUT_NAME}',
    #                 exp_name=f'BCP-switch-{args.mode}-{args.switch_human_freq}',
    #                 env_name='')

    for k in range(args.num_episodes):
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            # ai_act = evaluate(ai_agent, ai_obs)
            ai_act = ai_agent.choose_action(ai_obs, deterministic=True)
            h_act = bc_model.choose_action(h_obs, deterministic=True)
            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            env.render()
        # logger.update(score=[ep_reward], total_steps=k + 1)
        print(f'Ep {k+1}:',ep_reward)