import numpy as np
import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import BCP_MODELS, HP_MODELS
from bc.bc_hh import BehaviorClone
from utils import seed_everything, init_env
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
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ai_agent = torch.load(BCP_MODELS[args.layout], map_location='cpu')
    human_model = torch.load(HP_MODELS[args.layout], map_location='cpu')
    # now = datetime.now()
    # formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='BPR',
               name=f'BCP_{args.layout}_HP_seed{args.seed}',
               config=vars(args),
               job_type='eval',
               reinit=True)

    seed_everything(args.seed)
    env = init_env(layout=args.layout)
    for k in range(args.num_episodes):
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            ai_act = evaluate(ai_agent, ai_obs)
            h_act = human_model.choose_action(h_obs, deterministic=True)
            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            # env.render(interval=0.1)
        print(f'Ep {k+1}:',ep_reward)
        wandb.log({'episode': k+1, 'ep_reward': ep_reward})
    wandb.finish()


