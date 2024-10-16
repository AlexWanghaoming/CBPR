import numpy as np
import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import BCP_MODELS, SP_MODELS, FCP_MODELS, SKILL_MODELS
from My_utils import seed_everything, init_env, evaluate_actor, print_mean_interval
import wandb

WANDB_DIR = 'my_wandb_log'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--layout', default='soup_coordination')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--algorithm', default='SP', help='BCP or SP or FCP')
    parser.add_argument('--skill_level', default='high', help='low or medium or high')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.algorithm == 'BCP':
        ai_agent = torch.load(BCP_MODELS[args.layout], map_location='cpu')
    elif args.algorithm == 'FCP':
        ai_agent = torch.load(FCP_MODELS[args.layout], map_location='cpu')
    elif args.algorithm == 'SP':
        ai_agent = torch.load(SP_MODELS[args.layout], map_location='cpu')
    else:
        pass

    if args.skill_level == 'low':
        skill_model_path = SKILL_MODELS[args.layout][0]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    elif args.skill_level == 'medium':
        skill_model_path = SKILL_MODELS[args.layout][1]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    elif args.skill_level == 'high':
        skill_model_path = SKILL_MODELS[args.layout][2]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    else:
        pass
    if args.use_wandb:
        wandb.init(project='overcooked_rl',
                   group='exp2_2',
                   name=f'{args.algorithm}_{args.layout}_{args.skill_level}_seed{args.seed}',
                   config=vars(args),
                   job_type='eval',
                   dir=os.path.join(WANDB_DIR, 'exp2_2'),
                   reinit=True)

    seed_everything(args.seed)
    env = init_env(layout=args.layout)
    r_list = []
    for k in range(args.num_episodes):
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            ai_act = evaluate_actor(ai_agent, ai_obs, deterministic=True)
            h_act = evaluate_actor(skill_model, h_obs, deterministic=False)
            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            # obs, sparse_reward, done, info = env.step((h_act, ai_act))

            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            env.render(interval=0.05)
        print(f'Ep {k+1}:',ep_reward)
        r_list.append(ep_reward)
        if args.use_wandb:
            wandb.log({'episode': k+1, 'ep_reward': ep_reward})
    if args.use_wandb:
        wandb.finish()
    print_mean_interval(r_list)


