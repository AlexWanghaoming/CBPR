import numpy as np
import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import META_TASKS, BCP_MODELS, SP_MODELS, FCP_MODELS
from bc.bc_hh import BehaviorClone
from My_utils import seed_everything, init_env, evaluate_actor, print_mean_interval
import random
import wandb
from datetime import datetime

WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
HORIZON = 3000

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='coordination_ring')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--mode', default='inter', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=1, help="Frequency of switching human policy")
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--algorithm', default='BCP', help='BCP or SP or FCP')
    parser.add_argument('--use_wandb', action='store_true', default=False)

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
    if args.use_wandb:
        wandb.init(project='overcooked_rl',
                   group='exp1',
                   name=f'{args.algorithm}_{args.layout}_{args.mode}{args.switch_human_freq}_seed{args.seed}_horizon{HORIZON}',
                   config=vars(args),
                   job_type='eval',
                   dir=os.path.join(WANDB_DIR, 'exp1', args.algorithm),  # 这个目录需要手动创建
                   reinit=True)

    if args.mode == 'inter':
        random.seed(42) # 固定测试的策略顺序
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    if args.mode == 'intra':
        random.seed(42)
        N = args.num_episodes * (HORIZON // args.switch_human_freq)
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]

    seed_everything(args.seed)
    # 初始化人类模型和策略
    policy_idx = 2
    # print("初始策略: ", mts[policy_idx - 1])
    env = init_env(horizon=HORIZON,
                    layout=args.layout,
                   agent0_policy_name=args.algorithm,
                   agent1_policy_name=f'script:{mts[policy_idx-1]}',
                   use_script_policy=True)
    r_list = []
    for k in range(args.num_episodes):
        if args.mode == 'inter':
            # 每switch_human_freq 改变一次策略
            policy_idx = policy_id_list.pop()
            # print("人类改变至策略: ", mts[policy_idx-1]) #log
            env.switch_script_agent(agent0_policy_name=args.algorithm,
                                    agent1_policy_name=f'script:{mts[policy_idx-1]}')
        obs = env.reset()
        ai_obs, h_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            # print(episode_steps)
            if args.mode == "intra":
                # 轮内切换人的策略
                if episode_steps % args.switch_human_freq == 0:
                    policy_idx = policy_id_list.pop()
                    # print(f"人类改变至策略: ", mts[policy_idx-1])
                    env.switch_script_agent(agent0_policy_name=args.algorithm,
                                            agent1_policy_name=f'script:{mts[policy_idx - 1]}')
            ai_act = evaluate_actor(ai_agent, ai_obs)
            obs, sparse_reward, done, info = env.step((ai_act, 1))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            # env.render(interval=0.1)
        print(f'Ep {k+1}:',ep_reward)
        r_list.append(ep_reward)
        if args.use_wandb:
            wandb.log({'episode': k+1,
                       'ep_reward': ep_reward})
    print(f'{args.algorithm}_{args.layout}_{args.mode}_{args.switch_human_freq}')
    # print_mean_interval(r_list)
    if args.use_wandb:
        wandb.finish()


