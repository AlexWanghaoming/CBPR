import numpy as np
import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import META_TASKS, BCP_MODELS, SP_MODELS, FCP_MODELS, SKILL_MODELS
from bc.bc_hh import BehaviorClone
from My_utils import seed_everything, init_env, evaluate_actor, print_mean_interval
import random
import wandb
from datetime import datetime

WANDB_DIR = 'my_wandb_log'
device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='soup_coordination')
    # parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--algorithm', default='FCP', help='BCP or SP or FCP')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mts = META_TASKS[args.layout]
    if args.algorithm == 'BCP':
        ai_agent = torch.load(BCP_MODELS[args.layout], map_location='cpu')
    elif args.algorithm == 'FCP':
        ai_agent = torch.load(FCP_MODELS[args.layout], map_location='cpu')
    elif args.algorithm == 'SP':
        ai_agent = torch.load(SP_MODELS[args.layout], map_location='cpu')
    else:
        pass

    wandb.init(project='overcooked_rl',
               group='exp4',
               name=f'{args.algorithm}_{args.layout}_{args.mode}{args.switch_human_freq}_seed{args.seed}',
               config=vars(args),
               job_type='eval',
               dir=os.path.join(WANDB_DIR, 'exp4', args.algorithm),  # 这个目录需要手动创建
               reinit=True)

    if args.mode == 'inter':
        random.seed(42) # 固定测试的策略顺序
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, 3) for _ in range(N)]
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    if args.mode == 'intra':
        random.seed(42)
        N = args.num_episodes * (600 // args.switch_human_freq)
        policy_id_list = [random.randint(1, 3) for _ in range(N)]

    seed_everything(args.seed)
    # 初始化人类模型和策略
    policy_idx = 2
    # print("初始策略: ", mts[policy_idx - 1])
    env = init_env(layout=args.layout, use_script_policy=False)
    r_list = []
    skill_model_path = SKILL_MODELS[args.layout][policy_idx - 1]
    skill_model = torch.load(skill_model_path, map_location=device)
    for k in range(args.num_episodes):
        if args.mode == 'inter':
            # 每switch_human_freq 改变一次策略
            policy_idx = policy_id_list.pop()
            skill_model_path = SKILL_MODELS[args.layout][policy_idx - 1]
            skill_model = torch.load(skill_model_path, map_location=device)
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
                    skill_model_path = SKILL_MODELS[args.layout][policy_idx - 1]
                    skill_model = torch.load(skill_model_path, map_location=device)
            ai_act = evaluate_actor(ai_agent, ai_obs)
            h_act = evaluate_actor(skill_model, h_obs, deterministic=False, device=device)

            obs, sparse_reward, done, info = env.step((ai_act, h_act))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            # env.render(interval=0.1)
        # print(f'Ep {k+1}:',ep_reward)
        r_list.append(ep_reward)
        wandb.log({'episode': k+1, 'ep_reward': ep_reward})
    # print(f'{args.algorithm}_{args.layout}_{args.mode}_{args.switch_human_freq}')
    # print_mean_interval(r_list)
    wandb.finish()


