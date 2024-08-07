from copy import deepcopy
import numpy as np
from typing import *
import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import SKILL_MODELS, META_TASKS
from My_utils import seed_everything
import wandb
from algorithms.cbpr import CBPR

device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--skill_level', default='low', help='low or medium or high')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=600)
    parser.add_argument('--Q_len', type=int, default=20)
    parser.add_argument('--rho', type=float, default=0.1,
                        help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
    args = parse_args()
    if args.skill_level == 'low':
        skill_model_path = SKILL_MODELS[args.layout][0]
        skill_model = torch.load(skill_model_path, map_location=device)
    elif args.skill_level == 'medium':
        skill_model_path = SKILL_MODELS[args.layout][1]
        skill_model = torch.load(skill_model_path, map_location=device)
    elif args.skill_level == 'high':
        skill_model_path = SKILL_MODELS[args.layout][2]
        skill_model = torch.load(skill_model_path, map_location=device)
    else:
        pass

    n = len(META_TASKS[args.layout])
    if args.use_wandb:
        wandb.init(project='overcooked_rl',
                   group='exp2_3',
                   name=f'okr_{args.layout}_{args.skill_level}_seed{args.seed}_Q{args.Q_len}_rho{args.rho}',
                   # name=f'okr_{args.layout}_{args.skill_level}_seed{args.seed}_Q{args.Q_len}_rho{args.rho}_horizon{args.horizon}_{n}metatask',
                   config=vars(args),
                   job_type='eval',
                   # dir=os.path.join(WANDB_DIR, 'exp2', 'ablations'),
                   dir=os.path.join(WANDB_DIR, 'exp2_3'),
                   reinit=True)

    seed_everything(args.seed)
    bpr_online = CBPR(args)
    print('**----------------------------------------------------------------**')
    print(f'LAYOUT: {args.layout}')
    print(f'Start collaborating with agents using {args.skill_level} skill level')
    bpr_online.play(partner_policy=skill_model)
    if args.use_wandb:
        wandb.finish()






