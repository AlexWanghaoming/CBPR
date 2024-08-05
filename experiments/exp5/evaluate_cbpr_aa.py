import os
from algorithms.cbpr import CBPR
import wandb
import torch
from My_utils import seed_everything
from models import SP_MODELS, BCP_MODELS, FCP_MODELS, SP_ALT_MODELS
import argparse

WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
# args = cbpr.parse_args()
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Collaborative Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--layout', default='soup_coordination')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=600)
    parser.add_argument('--Q_len', type=int, default=20, help='length of behavio queue')
    parser.add_argument('--rho', type=float, default=0.1, help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args

args = parse_args()
if args.use_wandb:
    wandb.init(project='overcooked_rl',
               group='exp2',
               name=f'okr_{args.layout}_{args.skill_level}_seed{args.seed}_Q{args.Q_len}_rho{args.rho}_horizon{args.horizon}',
               config=vars(args),
               job_type='eval',
               dir=os.path.join(WANDB_DIR, ''),
               # dir=os.path.join(WANDB_DIR, 'exp2_2'),
               reinit=True)
seed_everything(args.seed)
bpr_online = CBPR(args)
# ai_agent = torch.load(SP_MODELS[args.layout], map_location='cpu')
# ai_agent = torch.load(BCP_MODELS[args.layout], map_location='cpu')
ai_agent = torch.load(FCP_MODELS[args.layout], map_location='cpu')

bpr_online.play(partner_policy=ai_agent)
if args.use_wandb:
    wandb.finish()

