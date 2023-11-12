from copy import deepcopy
import numpy as np
import re
from typing import *
import argparse
import torch
import torch.nn as nn
import os, sys
from models import MTP_MODELS, META_TASK_MODELS, NN_MODELS
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from agents.ppo_discrete import PPO_discrete
import random
from state_trans_func.NN_scriptedPolicy import NN
from utils import seed_everything, init_env
from datetime import datetime
from src.overcooked_ai_py.mdp.actions import Action
import wandb


device = 'cuda'
META_TASKS = ['place_onion_in_pot', 'deliver_soup', 'place_onion_and_deliver_soup', 'random']


class MetaTaskLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self) -> Dict[str, str]:
        for idx, mt in enumerate(META_TASKS):
            self.policy_lib[f'metatask_{idx+1}'] = mt
        return self.policy_lib

class MTPLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, args) -> Dict[str, PPO_discrete]:
        for idx, mtp_path in enumerate(MTP_MODELS[LAYOUT_NAME]):
            agent_id = f'mtp_{idx+1}'
            agent = PPO_discrete()
            agent.load_actor(mtp_path)
            # agent.load_critic(mtp_path[1])
            self.policy_lib[agent_id] = agent
        return self.policy_lib


class NNLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self) -> Dict[str, nn.Module]:
        for idx, bc_model_path in enumerate(NN_MODELS[LAYOUT_NAME]):
            state_dict = torch.load(bc_model_path)
            model = NN(input_dim=102, output_dim=97)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self.policy_lib[f'metatask_{idx+1}'] = model
        return self.policy_lib

# def eval_rollouts(env, ai_agent: PPO_discrete, h_policy:nn.Module):
#     obs = env.reset()
#     ai_obs, h_obs = obs['both_agent_obs']
#     accumulated_reward = 0
#     done = False
#     while not done:
#         # 策略库中的agent和人玩，得到rewards
#         # ai_act = agent_policy.predict(ai_obs)[0]
#         # ai_act = evaluate(agent_policy, ai_obs)  # 确定性策略
#         ai_act = ai_agent.evaluate(ai_obs)  # 随机策略
#         h_act = h_policy.choose_action(h_obs, deterministic=True)
#         obs, sparse_reward, done, info = env.step((ai_act,h_act))
#         ego_obs_, alt_obs_ = obs['both_agent_obs']
#         accumulated_reward += sparse_reward
#     return accumulated_reward


class BPR_offline:
    """
    BPR离线部分，performance model初始化，belief初始化
    """
    def __init__(self, args):
        HPL = MetaTaskLibrary()
        self.human_policys = HPL.gen_policy_library()
        self.ai_policys = APL.gen_policy_library(args)
        # print('init agent policy library is ', self.ai_policys)
        # print('init human policy library is ', self.human_policys)

    def gen_belief(self) -> Dict[str, float]:
        """
        init beta 均匀分布
        """
        lens = len(self.human_policys)
        assert lens > 0
        # beta = copy.deepcopy(self.human_policys)
        beta = self.human_policys
        for i in beta.keys():
            beta[i] = 1 / lens
        print('Initial belief is ', beta)
        ss = deepcopy(beta)
        return ss


class BPR_online:
    def __init__(self, agents: Dict[str, PPO_discrete],
                 human_policys: Optional[Dict[str, nn.Module]],
                 NN_models: Dict[str, nn.Module],
                 belief: Dict[str, float],
                 new_polcy_threshold=0.3):
        self.belief = belief
        self.mtp = agents
        self.meta_tasks = human_policys
        self.NNs = NN_models
        self.threshold = new_polcy_threshold
        self.eps = 1e-6

    def play(self, args, logger=None):
        args.max_episode_steps = 600
        total_steps = 0
        # 初始化人类模型和策略
        policy_idx = 2
        print("初始策略: ", META_TASKS[policy_idx - 1])
        env = init_env(layout=args.layout,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{META_TASKS[policy_idx - 1]}',
                       use_script_policy=True)
        for k in range(args.num_episodes):
            if args.mode == 'inter':
                policy_idx = policy_id_list.pop()
                print("人类改变至策略: ", META_TASKS[policy_idx - 1])  # log
                env.switch_script_agent(agent0_policy_name='mtp',
                                        agent1_policy_name=f'script:{META_TASKS[policy_idx - 1]}')
            episode_steps = 0
            best_agent_id_prime = ""
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            # best_agent_id, best_agent = self._reuse_optimal_policy()
            while not done:
                best_agent_id, best_agent = self._reuse_optimal_policy()  # 选择belief最大的智能体
                total_steps += 1
                episode_steps += 1
                if args.mode == "intra":
                    if episode_steps % args.switch_human_freq == 0:
                        policy_idx = policy_id_list.pop()
                        print("人类改变至策略: ", META_TASKS[policy_idx - 1])  # log
                        env.switch_script_agent(agent0_policy_name='bcp',
                                                agent1_policy_name=f'script:{META_TASKS[policy_idx - 1]}')
                ai_act = best_agent.evaluate(ai_obs)  # 智能体选动作
                obs_, sparse_reward, done, info = env.step((ai_act, 1))
                ep_reward += sparse_reward
                ai_obs_, h_obs_ = obs_['both_agent_obs']
                h_dire = info['joint_action'][1]
                h_act = Action.INDEX_TO_ACTION.index(h_dire)
                # wanghm 用NN作预测
                actions_one_hot = np.eye(6)[h_act]
                obs_x = np.hstack([h_obs, actions_one_hot])
                obs_x = torch.from_numpy(obs_x).float().to(device)
                obs_y = np.hstack([h_obs_, sparse_reward]) # s_prime, r
                # obs_y = h_obs_ # s_prime
                obs_y = torch.from_numpy(obs_y).float().to(device)

                self.belief = self._update_beta(obs_x=obs_x,
                                                obs_y=obs_y)
                ai_obs, h_obs = ai_obs_, h_obs_

                # # debug: 直接选对应的策略作为最优策略
                # best_agent_id = list(self.mtp.keys())[policy_idx - 1]
                # best_agent = self.mtp[best_agent_id]

                # # log
                # if best_agent_id != best_agent_id_prime:
                #     print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                #     best_agent_id_prime = best_agent_id

            print(f'Ep {k + 1} rewards: {ep_reward}')
            wandb.log({'episode': k+1, 'ep_reward': ep_reward})

    def _update_beta(self, obs_x:torch.float, obs_y:torch.float) -> Dict[str, float]:
        """
        每 episode 利用Bayesian公式更新回合间belif
        belief: dict
        """
        p_temp = {}
        new_belief = {}
        eps = self.eps
        for id in self.belief:
            NN = self.NNs[id]
            means = NN(obs_x)
            vars = torch.tensor([0.1]).to(device)
            std_devs = torch.sqrt(vars)
            probs = torch.exp(-0.5*((obs_y - means) / std_devs)**2) / (std_devs * torch.sqrt(2 * torch.tensor([3.141592653589793]).to(device)))
            p_temp[id] = torch.mean(probs).item() * self.belief[id]
        for id in self.belief:
            new_belief[id] = (p_temp[id] + eps) / (sum(p_temp.values()))
        # print(new_belief) # log
        return new_belief

    def _reuse_optimal_policy(self) -> Tuple[str, PPO_discrete]:
        """
        calculate expectation of utility
        用BPR-EI求最优策略

        return: 最优策略id，最优策略
        """

        # u_mean_x_list = []
        # for ai in self.mtp:
        #     temp = 0
        #     for h in self.meta_tasks:
        #         gp = self.gps[h]
        #         temp += belief[h] * gp()
        #     u_mean_x_list.append(temp)
        # u_mean = max(u_mean_x_list)
        # # u_mean = 0
        # # get the upper value of utility
        # u_upper = self._get_u_upper()
        # # 求积分
        # delta_u = (u_upper - u_mean) / 1000
        # u_list = np.linspace(u_mean, u_upper, 1000)
        # likely_list = {}
        #
        # # wanghm: time consuming !
        # for ai_ in self.mtp:
        #     inner = 0
        #     for u_ in u_list:
        #         for h_ in self.meta_tasks:
        #             # inner += self.belief[h_] * self._gen_pdf(self.performance_model[0][h_][ai_],
        #             #                                          self.performance_model[1][h_][ai_],
        #             #                                          u_) * delta_u * u_
        #             inner += belief[h_] * self._gen_pdf(self.performance_model[0][h_][ai_],
        #                                                 self.performance_model[1][h_][ai_],
        #                                                 u_) * delta_u
        #     likely_list[ai_] = inner

        # best_agent_name = max(self.belief, key=self.belief.get)
        # best_agent = self.mtp[max(self.belief, key=self.belief.get)]
        target_task = max(self.belief, key=self.belief.get) # "metatask_1"
        idx = target_task.split('_')[-1]
        best_agent_name = 'mtp_' + idx # "mtp_1"
        best_agent = self.mtp[best_agent_name]
        return best_agent_name, best_agent

    # def _get_u_upper(self):
    #     """
    #     Define: U_max =  u + 3*sigma
    #     """
    #     upper_list = []
    #     for h_ in self.human_policys:
    #         for ai_ in self.agents:
    #             upper_list.append(self.performance_model[0][h_][ai_] + 3 * self.performance_model[1][h_][ai_])
    #     upper = max(upper_list)
    #     return upper


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    LAYOUT_NAME = args.layout
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='BPR',
               name=f'bprNN_{args.layout}_{args.mode}{args.switch_human_freq}_{formatted_now}',
               config=vars(args),
               job_type='eval',
               reinit=True)

    if args.mode == 'inter':
        random.seed(42)
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, 4) for _ in range(N)]  # 固定测试的策略顺序
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    else:
        random.seed(42)
        N = args.num_episodes * (600 // args.switch_human_freq)
        policy_id_list = [random.randint(1, 4) for _ in range(N)]

    seed_everything(args.seed)
    APL = MTPLibrary()
    mtp_lib = APL.gen_policy_library(args)  # 构建AI策略库
    NNL = NNLibrary()
    NN_models = NNL.gen_policy_library()
    bpr_offline = BPR_offline(args)
    belief = bpr_offline.gen_belief()
    bpr_online = BPR_online(agents=mtp_lib,
                            # human_policys=meta_task_lib,
                            human_policys=None,
                            NN_models=NN_models,
                            belief=belief)

    bpr_online.play(args, logger=None)
    wandb.finish()






