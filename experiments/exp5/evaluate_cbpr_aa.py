import os
from algorithms import cbpr
import wandb
import torch
from My_utils import seed_everything
from models import SP_MODELS, BCP_MODELS, FCP_MODELS


WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
args = cbpr.parse_args()

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
bpr_online = cbpr.BPR_online(args)
# ai_agent = torch.load(SP_MODELS[args.layout], map_location='cpu')
ai_agent = torch.load(BCP_MODELS[args.layout], map_location='cpu')
# ai_agent = torch.load(FCP_MODELS[args.layout], map_location='cpu')

bpr_online.play(partner_policy=ai_agent)
if args.use_wandb:
    wandb.finish()