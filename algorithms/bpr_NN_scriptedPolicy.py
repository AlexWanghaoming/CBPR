from copy import deepcopy
import numpy as np
from collections import deque
from typing import *
import argparse
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from bc.bc_hh import BehaviorClone
from models import MTP_MODELS, NN_MODELS, META_TASKS
from agents.ppo_discrete import PPO_discrete
import random
from state_trans_func.NN_scriptedPolicy import NN
from My_utils import seed_everything, init_env
from datetime import datetime
from src.overcooked_ai_py.mdp.actions import Action
import wandb


device = 'cuda'


class MetaTaskLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, tasks) -> Dict[str, str]:
        for idx, mt in enumerate(tasks):
            self.policy_lib[f'metatask_{idx+1}'] = mt
        return self.policy_lib

class MTPLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, args) -> Dict[str, PPO_discrete]:
        for idx, mtp_path in enumerate(MTP_MODELS[args.layout]):
            agent_id = f'mtp_{idx+1}'
            agent = PPO_discrete()
            agent.load_actor(mtp_path)
            # agent.load_critic(mtp_path[1])
            self.policy_lib[agent_id] = agent
        return self.policy_lib


class NNLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, args) -> Dict[str, nn.Module]:
        for idx, bc_model_path in enumerate(NN_MODELS[args.layout]):
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
        APL = MTPLibrary()
        self.human_policys = HPL.gen_policy_library(META_TASKS[args.layout])
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
        mts = META_TASKS[args.layout]
        print("初始策略: ", mts[policy_idx - 1])
        env = init_env(layout=args.layout,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{mts[policy_idx - 1]}',
                       use_script_policy=True)
        # n_hit = 0
        for k in range(args.num_episodes):
            if args.mode == 'inter':
                policy_idx = policy_id_list.pop()
                print("人类改变至策略: ", mts[policy_idx - 1])  # log
                env.switch_script_agent(agent0_policy_name='mtp',
                                        agent1_policy_name=f'script:{mts[policy_idx - 1]}')
            Q = deque(maxlen=args.Q_len)  # 记录maxlen条人类交互数据 (s,a,s',r)

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
                        print("人类改变至策略: ", mts[policy_idx - 1])  # log
                        env.switch_script_agent(agent0_policy_name='bcp',
                                                agent1_policy_name=f'script:{mts[policy_idx - 1]}')
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
                Q.append((obs_x, obs_y))

                self.belief = self._update_beta(Q)
                # if hit:
                #     n_hit += 1
                # print(f'Accuracy: {n_hit}/{total_steps}')

                ai_obs, h_obs = ai_obs_, h_obs_

                # # debug: 直接选对应的策略作为最优策略
                # best_agent_id = list(self.mtp.keys())[policy_idx - 1]
                # best_agent = self.mtp[best_agent_id]
                # # log
                # if best_agent_id != best_agent_id_prime:
                #     print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                #     best_agent_id_prime = best_agent_id

            print(f'Ep {k + 1} rewards: {ep_reward}')
            # wandb.log({'episode': k+1, 'ep_reward': ep_reward})

    def _update_beta(self,Q) -> Dict[str, float]:
        """
        这里和BPR NN *Efficient Bayesian Policy Reuse With a Scalable Observation Model in Deep Reinforcement Learning, TNNLS* 有区别
        belief: dict
        """
        Q_prob = {}
        pi = torch.tensor([3.141592653589793]).to(device)
        var = torch.tensor([0.1]).to(device)
        std_dev = torch.sqrt(var)
        denominator = std_dev * torch.sqrt(2 * pi)

        temp = {}
        new_belief = {}
        eps = self.eps
        exp_v = []

        for mt in self.belief:
            NN = self.NNs[mt]
            su = 0
            for s_a, s_primer_r in Q:
                means = NN(s_a)
                probs = torch.exp(-0.5 * ((s_primer_r - means) / std_dev) ** 2) / denominator
                su += torch.log(torch.mean(probs)).item()
            # exp_v.append(su)
            temp[mt] = np.exp(su) * self.belief[mt]

        # max_index = max(enumerate(exp_v), key=lambda x: x[1])[0] + 1
        total = sum(temp.values())
        for mt in self.belief:
            new_belief[mt] = (temp[mt] + eps) / total

        # return new_belief, max_index == policy_idx

        return new_belief

    def _reuse_optimal_policy(self) -> Tuple[str, PPO_discrete]:
        target_task = max(self.belief, key=self.belief.get) # "metatask_1"
        idx = target_task.split('_')[-1]
        best_agent_name = 'mtp_' + idx             # "mtp_1"
        best_agent = self.mtp[best_agent_name]
        return best_agent_name, best_agent


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--mode', type=str, default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    parser.add_argument('--seed', type=int, default=0),
    parser.add_argument('--Q_len', type=int, default=25)
    parser.add_argument('--plot_group', type=str, default='BPR_NN_MOD')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    LAYOUT_NAME = args.layout
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    # wandb.init(project='overcooked_rl',
    #            group='BPR',
    #            name=f'bprNN_{args.layout}_{args.mode}{args.switch_human_freq}_seed{args.seed}',
    #            config=vars(args),
    #            job_type='eval',
    #            reinit=True)

    if args.mode == 'inter':
        random.seed(42)
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]  # 固定测试的策略顺序
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    else:
        random.seed(42)
        N = args.num_episodes * (600 // args.switch_human_freq)
        policy_id_list = [random.randint(1, len(META_TASKS[args.layout])) for _ in range(N)]

    seed_everything(args.seed)
    APL = MTPLibrary()
    mtp_lib = APL.gen_policy_library(args)  # 构建AI策略库
    NNL = NNLibrary()
    NN_models = NNL.gen_policy_library(args)
    bpr_offline = BPR_offline(args)
    belief = bpr_offline.gen_belief()
    bpr_online = BPR_online(agents=mtp_lib,
                            # human_policys=meta_task_lib,
                            human_policys=None,
                            NN_models=NN_models,
                            belief=belief)

    bpr_online.play(args, logger=None)
    # wandb.finish()






