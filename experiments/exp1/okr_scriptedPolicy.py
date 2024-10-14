from copy import deepcopy
from collections import deque
import numpy as np
from typing import List, Tuple, Dict
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import MTP_MODELS, METATASK_MODELS, META_TASKS
from agents.ppo_discrete import PPO_discrete
from My_utils import seed_everything, init_env, limit_value
import random
from algorithms.bc.opponent_scriptedPolicy import Opponent
from algorithms.bc.bc_hh import BehaviorClone
import torch.nn as nn
import torch
import wandb
from src.overcooked_ai_py.mdp.actions import Action
from algorithms.cbpr import CBPR

device = 'cpu'
HORIZON = 600

class CBPR_switch(CBPR):
    def __init__(self, args):
        super(CBPR_switch, self).__init__(args)

    def play(self):
        """
        rewrite play method to collaborate with partners that switch their policy.
        """
        self.args.max_episode_steps = HORIZON
        total_steps = 0
        # init human policy
        policy_idx = 2
        print(f"初始策略 metatask_{policy_idx}: ", META_TASKS[self.args.layout][policy_idx - 1])
        env = init_env(horizon=HORIZON,
                       layout=self.args.layout,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{META_TASKS[self.args.layout][policy_idx - 1]}',
                       use_script_policy=True)
        for k in range(self.args.num_episodes):
            if self.args.mode == 'inter':
                policy_idx = policy_id_list.pop()
                # print(f"人类改变至策略 metatask_{policy_idx}: ", META_TASKS[self.args.layout][policy_idx - 1])  # log
                env.switch_script_agent(agent0_policy_name='bcp',
                                        agent1_policy_name=f'script:{META_TASKS[self.args.layout][policy_idx - 1]}')
            Q = deque(maxlen=self.args.Q_len)
            episode_steps = 0
            self.xi = deepcopy(self.belief)
            best_agent_id, best_agent = self._reuse_optimal_policy(belief=self.belief)  # 选择初始智能体策略
            best_agent_id_prime = ""
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            while not done:
                total_steps += 1
                episode_steps += 1
                if self.args.mode == "intra":
                    # 轮内切换人的策略
                    if episode_steps % self.args.switch_human_freq == 0:
                        policy_idx = policy_id_list.pop()
                        # print(f"人类改变至策略 metatask_{policy_idx}: ", META_TASKS[self.args.layout][policy_idx - 1])  # log
                        env.switch_script_agent(agent0_policy_name='mtp',
                                                agent1_policy_name=f'script:{META_TASKS[self.args.layout][policy_idx - 1]}')
                # ai_act = best_agent.evaluate(ai_obs)
                ai_act, _ = best_agent.choose_action(ai_obs)
                # h_act = bc_model.choose_action(h_obs)
                obs_, sparse_reward, done, info = env.step((ai_act, 1))
                ep_reward += sparse_reward
                ai_obs, h_obs = obs_['both_agent_obs']
                h_dire = info['joint_action'][1]
                h_act = Action.INDEX_TO_ACTION.index(h_dire)

                h_obs = torch.tensor(h_obs, dtype=torch.float32).to(device)
                h_act = torch.tensor(np.array([h_act]), dtype=torch.int64).to(device)
                Q.append((h_obs, h_act))

                # if episode_steps % 1 == 0 and len(Q) == self.args.Q_len:
                self.xi = self._update_xi(Q)  # 更新intra-episode belief $\xi$ 原文公式8,9
                # print('xi: ', self.xi)
                self.zeta = self._update_zeta(t=episode_steps, rho=self.args.rho)  # 更新integrated belief $\zeta$ 原文公式10
                best_agent_id, best_agent = self._reuse_optimal_policy(belief=self.zeta)  # 轮内重用最优策略
                self.xi = deepcopy(self.zeta)  # 每一步更新 \xi

                # best_agent_id = list(self.agents.keys())[policy_idx-1]
                # best_agent = self.agents[best_agent_id]
                # if best_agent_id != best_agent_id_prime:
                #     print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                #     best_agent_id_prime = best_agent_id
            print(f'Ep {k + 1} rewards: {ep_reward}')
            wandb.log({'episode': k+1, 'ep_reward': ep_reward})

            # update belief of current episode
            self.belief = deepcopy(self.xi)
            self.belief = self._update_beta(best_agent_id, ep_reward)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--mode', default='inter', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=1, help="Frequency of switching human policy")
    parser.add_argument('--Q_len', type=int, default=5)
    parser.add_argument('--rho', type=float, default=0.1, help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument('--horizon', type=int, default=600)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    WANDB_DIR = 'my_wandb_log'
    args = parse_args()
    LAYOUT_NAME = args.layout
    wandb.init(project='overcooked_rl',
               group='exp1',
               name=f'okr_{args.layout}_{args.mode}{args.switch_human_freq}_seed{args.seed}_horizon{HORIZON}',
               config=vars(args),
               job_type='eval',
               dir=os.path.join(WANDB_DIR, 'exp1', 'okr'),
               reinit=True)

    if args.mode == 'inter':
        random.seed(42)
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]  # 固定测试的策略顺序
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    if args.mode == 'intra':
        random.seed(42)
        N = args.num_episodes * (HORIZON // args.switch_human_freq)
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]

    seed_everything(args.seed)
    bpr_online = CBPR_switch(args)
    bpr_online.play()

    wandb.finish()







