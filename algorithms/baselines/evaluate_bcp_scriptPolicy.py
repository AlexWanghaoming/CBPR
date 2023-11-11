import numpy as np
import argparse
import torch
import os, sys
from models import BCP_MODELS
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
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ai_agent = torch.load(BCP_MODELS[args.layout])
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    # wandb.init(project='overcooked_rl',
    #            group='bpr_NN',
    #            name=f'BCP_{args.layout}_{args.mode}-{args.switch_human_freq}_{formatted_now}',
    #            config=vars(args),
    #            job_type='eval',
    #            reinit=True)
    if args.mode == 'inter':
        random.seed(42) # 固定测试的策略顺序
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, 4) for _ in range(N)]
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    if args.mode == 'intra':
        random.seed(42)
        N = args.num_episodes * (600 // args.switch_human_freq)
        policy_id_list = [random.randint(1, 4) for _ in range(N)]

    seed_everything(args.seed)
    # 初始化人类模型和策略
    policy_idx = 2
    meta_tasks = ['place_onion_in_pot', 'deliver_soup', 'random', 'place_onion_and_deliver_soup']
    print("初始策略: ", meta_tasks[policy_idx - 1])
    env = init_env(layout=args.layout,
                   agent0_policy_name='bcp',
                   agent1_policy_name=f'script:{meta_tasks[policy_idx-1]}',
                   use_script_policy=True)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    for k in range(args.num_episodes):
        if args.mode == 'inter':
            # 每switch_human_freq 改变一次策略
            policy_idx = policy_id_list.pop()
            print("人类改变至策略: ", meta_tasks[policy_idx-1]) #log
            env.switch_script_agent(agent0_policy_name='bcp',
                                    agent1_policy_name=f'script:{meta_tasks[policy_idx-1]}')
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
                    print(f"人类改变至策略: ", meta_tasks[policy_idx-1])
                    env.switch_script_agent(agent0_policy_name='bcp',
                                            agent1_policy_name=f'script:{meta_tasks[policy_idx - 1]}')
            ai_act = evaluate(ai_agent, ai_obs)
            obs, sparse_reward, done, info = env.step((ai_act, 1))
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward += sparse_reward
            env.render(interval=0.1)
        print(f'Ep {k+1}:',ep_reward)
        # wandb.log({'episode': k+1, 'ep_reward': ep_reward})
    # wandb.finish()


