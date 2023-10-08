from copy import deepcopy
from collections import deque
import math
import numpy as np
import re, pickle
from typing import List, Tuple, Dict, Optional, Type, Union
import argparse
import torch
import torch.nn as nn

import os, sys
from models import MTP_MODELS, META_TASK_MODELS
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from bc.bc_hh import BehaviorClone
from agents.ppo_discrete import PPO_discrete
from utils import seed_everything, LinearAnnealer, init_env
from rl_plotter.logger import Logger
import random


class HumanPolicyLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self) -> Dict[str, nn.Module]:
        for bc_model_path in META_TASK_MODELS[LAYOUT_NAME]:
            pattern = r'\(\d+\.\d+(?:, \d+\.\d+)*\)'
            user_name = re.findall(pattern, bc_model_path)[0]
            policy = torch.load(bc_model_path)
            self.policy_lib[user_name] = policy
        return self.policy_lib


class AiPolicyLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, args) -> Dict[str, PPO_discrete]:
        for ppo_bc_model_path in MTP_MODELS[LAYOUT_NAME]:
            pattern = r'\(\d+\.\d+(?:, \d+\.\d+)*\)'
            agent_id = re.findall(pattern, ppo_bc_model_path)[0]

            agent = PPO_discrete(args)
            agent.load_actor(ppo_bc_model_path)
            # agent.load_critic(ppo_bc_model_path[1])

            self.policy_lib[agent_id] = agent
        return self.policy_lib


def eval_rollouts(env, ai_agent: PPO_discrete, h_policy:nn.Module):
    obs = env.reset()
    ai_obs, h_obs = obs['both_agent_obs']
    accumulated_reward = 0
    done = False
    while not done:
        # 策略库中的agent和人玩，得到rewards
        # ai_act = agent_policy.predict(ai_obs)[0]
        # ai_act = evaluate(agent_policy, ai_obs)  # 确定性策略
        ai_act = ai_agent.evaluate(ai_obs)  # 随机策略
        h_act = h_policy.choose_action(h_obs, deterministic=True)
        obs, sparse_reward, done, info = env.step((ai_act,h_act))
        ego_obs_, alt_obs_ = obs['both_agent_obs']
        accumulated_reward += sparse_reward
    return accumulated_reward


#####################################################
class BPR_offline:
    """
    BPR离线部分，performance model初始化，belief初始化
    """

    def __init__(self, args):
        # policy name(index) library
        self.human_policys = HPL.gen_policy_library()
        self.ai_policys = APL.gen_policy_library(args)
        # print('init agent policy library is ', self.ai_policys)
        # print('init human policy library is ', self.human_policys)

    def gen_performance_model(self) -> List[Dict[str, Dict[str, float]]]:
        """
        生成performance model
        implementation of performance model: [
                                              {'p1':{ppo-bc_p1: , 'ppo-bc_p2: ,  , '}}, 'p2':{ppo-bc_p1: , 'ppo-bc_p2: ,  , '}},
                                              {'p1':{ppo-bc_p1: , 'ppo-bc_p2: ,  , '}}, 'p2':{ppo-bc_p1: , 'ppo-bc_p2: ,  , '}}
                                              ]
        """

        performance_model_save_path = f'../models/performance/init_performance_{LAYOUT_NAME}.pkl'
        if os.path.exists(performance_model_save_path):
            with open(performance_model_save_path, 'rb') as ff:
                performance_model = pickle.load(ff)
            return performance_model

        env = init_env(layout=args.layout, lossless_state_encoding=False)

        performance_model = [{}, {}]
        n_rounds = 50  # TODO: performance model n_rounds当前是玩20次的平均值
        for h_ in self.human_policys:
            temp_mean = {}
            temp_std = {}

            for ai_ in self.ai_policys:
                ai_policy = self.ai_policys[ai_]
                h_policy = self.human_policys[h_]
                u_list = []
                for _ in range(n_rounds):
                    episodic_reward = eval_rollouts(env, ai_policy, h_policy)
                    u_list.append(episodic_reward)
                temp_mean[ai_] = np.mean(u_list)
                temp_std[ai_] = np.std(u_list)
            performance_model[0][h_] = temp_mean
            performance_model[1][h_] = temp_std

        with open(performance_model_save_path, 'wb') as f:
            pickle.dump(performance_model, f)

        return performance_model

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
                 human_policys: Dict[str, nn.Module],
                 performance_model: List[Dict[str, Dict[str, float]]],
                 belief: Dict[str, float],
                 new_polcy_threshold=0.3):
        self.performance_model = performance_model
        self.belief = belief
        self.agents = agents
        self.human_policys = human_policys
        self.threshold = new_polcy_threshold
        self.eps = 5e-6

        self.env = init_env(layout=LAYOUT_NAME, lossless_state_encoding=False)

    def update_performance_model(self, cur_agent_id, cur_agent):
        n_rounds = 20  # TODO: performance model n_rounds当前是玩20次的平均值
        for h_ in self.human_policys:
            # ego_idx = int(h_.replace('p', ''))
            h_policy = self.human_policys[h_]
            u_list = []
            for _ in range(n_rounds):
                episodic_reward = eval_rollouts(self.env, cur_agent, h_policy)
                u_list.append(episodic_reward)
            temp_mean = np.mean(u_list)
            temp_std = np.std(u_list)
            self.performance_model[0][h_][cur_agent_id] = temp_mean
            self.performance_model[1][h_][cur_agent_id] = temp_std

        performance_model_save_path = f'../models/performance/updated_performance_{LAYOUT_NAME}.pkl'
        with open(performance_model_save_path, 'wb') as f:
            pickle.dump(self.performance_model, f)
        return performance_model

    def play(self, args, logger):
        args.max_episode_steps = 600

        # buffer_dict = {"ppo-bc_p1": ReplayBuffer(args),
        #                "ppo-bc_p2": ReplayBuffer(args),
        #                "ppo-bc_p3": ReplayBuffer(args),
        #                "ppo-bc_p4": ReplayBuffer(args)
        #                }
        total_steps = 0

        # 初始化人类模型和策略
        policy_idx = 1
        bc_model = bc_models[policy_idx - 1]
        env = init_env(layout=args.layout, lossless_state_encoding=False)
        for k in range(args.num_episodes):
            if args.mode == 'inter':
                # 每switch_human_freq 改变一次策略
                policy_idx = policy_id_list.pop()
                # print("人类改变至策略: ", policy_idx) # debug
                bc_model = bc_models[policy_idx - 1]
            Q = deque(maxlen=25)  # 记录maxlen条人类交互数据 (s, a)
            episode_steps = 0
            self.xi = deepcopy(self.belief)
            best_agent_id, best_agent = self._reuse_optimal_policy()  # 选择初始智能体策略
            best_agent_id_prime = ""
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            while not done:
                total_steps += 1
                episode_steps += 1

                if args.mode == "intra":
                    # 轮内切换人的策略
                    if episode_steps % args.switch_human_freq == 0:
                        policy_idx = policy_id_list.pop()
                        # print(f"人类改变至策略:  {policy_idx}")
                        bc_model = bc_models[policy_idx - 1]

                ai_act = best_agent.evaluate(ai_obs)
                h_act = bc_model.choose_action(h_obs)
                # if args.new_policy_learning:
                #     _, logprob = best_agent.choose_action(ai_obs)

                obs_, sparse_reward, done, info = env.step((ai_act, h_act))
                ep_reward += sparse_reward
                ai_obs_, h_obs_ = obs_['both_agent_obs']
                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                # if args.new_policy_learning:
                #     replaybuffer = buffer_dict[best_agent_id]  # 每个智能体的 replay buffer存储各自的交互数据
                #     replaybuffer.store(ai_obs, ai_act, logprob, r, ai_obs_, dw, done)
                #     if replaybuffer.count == args.batch_size:
                #         best_agent.update(replaybuffer, total_steps)  # 自适应优化智能体
                #         replaybuffer.count = 0
                #         # print(f'更新智能体{best_agent_id}, 更新性能模型')
                #         self.update_performance_model(best_agent_id, best_agent)
                #         # print("Current performance model: ", self.performance_model)

                ai_obs = ai_obs_
                h_obs = h_obs_
                Q.append((h_obs, h_act))  # 将当前时间步的人类（s,a）加入经验池
                self.xi = self._update_xi(self.human_policys, Q)  # 更新intra-episode belief \xi
                # print('xi: ', self.xi)
                self.zeta = self._update_zeta(t=episode_steps, rho=0.5)  # 更新integrated belief \zeta
                # print('zeta: ', self.zeta)

                ### debug: 直接选对应的策略作为最优策略
                best_agent_id = list(self.agents.keys())[policy_idx-1]
                best_agent = self.agents[best_agent_id]

                # best_agent_id, best_agent = self._reuse_optimal_policy(
                #     inter_episode=True if args.mode == 'inter' else False)  # 轮内重用最优策略

                # debug
                if best_agent_id != best_agent_id_prime:
                    # print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                    best_agent_id_prime = best_agent_id

                self.xi = deepcopy(self.zeta)  # 每一步更新 \xi
                # episode_steps += 1
                # env.render()

            print(f'Ep {k + 1} rewards: {ep_reward}')
            logger.update(score=[ep_reward], total_steps=k + 1)
            # print("----------------------------------------------------------------------------------------------------")
            # 更新本轮的belief
            self.belief = deepcopy(self.xi)
            self.belief = self._update_beta(best_agent_id, ep_reward)

        # return rewards_list

    def _update_zeta(self, t, rho):
        new_zeta = {}
        for id in self.belief:
            new_zeta[id] = (rho ** t) * self.belief[id] + (1 - rho ** t) * self.xi[id]
            # new_zeta[id] = (rho/t) * self.belief[id] + (1- rho/t) *self.xi[id]

        return new_zeta

    def _update_beta(self, agent_id: str, u: float) -> Dict[str, float]:
        """
        每 episode 利用Bayesian公式更新回合间belif
        belief: dict
        """
        p_temp = {}
        new_belief = {}
        eps = self.eps
        for id in self.belief:
            p_temp[id] = self._gen_pdf(self.performance_model[0][id][agent_id],
                                       self.performance_model[1][id][agent_id],
                                       u) * self.belief[id]
        for id in self.belief:
            new_belief[id] = (p_temp[id] + eps) / (sum(p_temp.values()) + eps * 4)

        return new_belief

    def _update_xi(self, human_policy, Q):
        """
        每 step 利用Bayesian公式更新回合内belif
        xi: dict
        """

        def gen_Q_prob(Q):
            Q_prob = {}
            temp = {}
            eps = self.eps
            for id in self.xi:
                su = 0
                for q in Q:
                    s, a = q
                    su += np.log(human_policy[id].action_probability(s)[a])
                temp[id] = np.exp(su)
            for id in self.xi:
                Q_prob[id] = (temp[id] + eps) / (sum(temp.values()) + eps * 4)

            return Q_prob

        p_temp = {}
        new_xi = {}
        eps = self.eps
        Q_prob = gen_Q_prob(Q)
        for id in self.xi:
            p_temp[id] = Q_prob[id] * self.xi[id]
        for id in self.xi:
            new_xi[id] = (p_temp[id] + eps) / (sum(p_temp.values()) + eps * 4)

        return new_xi

    def _gen_pdf(self, mu, sigma, x):
        sigma = sigma + self.eps
        return (1 / (sigma * (math.pi * 2) ** 0.5)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def _reuse_optimal_policy(self, inter_episode=True) -> Tuple[str, PPO_discrete]:
        """
        calculate expectation of utility
        用BPR-EI求最优策略

        return: 最优策略id，最优策略
        """
        if inter_episode:
            belief = self.belief
        else:
            belief = self.zeta

        u_mean_x_list = []
        for ai in self.agents:
            temp = 0
            for h in self.human_policys:
                temp += belief[h] * self.performance_model[0][h][ai]
            u_mean_x_list.append(temp)
        u_mean = max(u_mean_x_list)
        # u_mean = 0
        # get the upper value of utility
        u_upper = self._get_u_upper()
        # 求积分
        delta_u = (u_upper - u_mean) / 1000
        u_list = np.linspace(u_mean, u_upper, 1000)
        likely_list = {}

        # wanghm: time consuming !
        for ai_ in self.agents:
            inner = 0
            for u_ in u_list:
                for h_ in self.human_policys:
                    # inner += self.belief[h_] * self._gen_pdf(self.performance_model[0][h_][ai_],
                    #                                          self.performance_model[1][h_][ai_],
                    #                                          u_) * delta_u * u_
                    inner += belief[h_] * self._gen_pdf(self.performance_model[0][h_][ai_],
                                                        self.performance_model[1][h_][ai_],
                                                        u_) * delta_u
            likely_list[ai_] = inner
        best_agent_id, best_agent = max(likely_list, key=likely_list.get), \
            self.agents[max(likely_list, key=likely_list.get)]
        return best_agent_id, best_agent

    def _get_u_upper(self):
        """
        Define: U_max =  u + 3*sigma
        """
        upper_list = []
        for h_ in self.human_policys:
            for ai_ in self.agents:
                upper_list.append(self.performance_model[0][h_][ai_] + 3 * self.performance_model[1][h_][ai_])
        upper = max(upper_list)
        return upper


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument("--net_arch", type=str, default='mlp', help="policy net arch")
    parser.add_argument("--hidden_width", type=int, default=32,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: 学习率线性衰减")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip: 0.1")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="value function coeffcient")

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='marshmallow_experiment', help='layout name')

    parser.add_argument('--num_episodes', type=int, default=200, help='total episodes')
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument('--new_policy_learning', default=False, action='store_true', help='whether to learn new policy')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    test_env = init_env(layout=args.layout, lossless_state_encoding=False)
    if args.net_arch == "conv":
        args.state_dim = test_env.observation_space.shape[-1]
    else:
        args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n

    LAYOUT_NAME = args.layout

    bc_models = []
    for i in range(4):
        bc_models.append(torch.load(META_TASK_MODELS[args.layout][i]))  # wanghm GAIL载入模型时不释放内存，需要提前存到列表

    seeds = [0, 1, 42, 2022, 2023]
    # seeds = [0, 1, 42]

    for seed in seeds:
        if args.mode == 'inter':
            random.seed(55)
            N = args.num_episodes // args.switch_human_freq
            policy_id_list = [random.randint(1, 4) for _ in range(N)]  # 固定测试的策略顺序
            policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
        else:
            random.seed(55)
            N = args.num_episodes * (600 // args.switch_human_freq)
            policy_id_list = [random.randint(1, 4) for _ in range(N)]

        seed_everything(seed)
        HPL = HumanPolicyLibrary()
        h_pl = HPL.gen_policy_library()  # 构建Human策略库
        APL = AiPolicyLibrary()
        apl = APL.gen_policy_library(args)  # 构建AI策略库

        bpr_offline = BPR_offline(args)

        performance_model = bpr_offline.gen_performance_model()
        print("Initial performance model: ", performance_model)
        belief = bpr_offline.gen_belief()
        bpr_online = BPR_online(agents=APL.gen_policy_library(args),
                                human_policys=HPL.gen_policy_library(),
                                performance_model=performance_model,
                                belief=belief,
                                new_polcy_threshold=0.3)

        if args.new_policy_learning:
            logger = Logger(log_dir=f'./logs/cbpr/{LAYOUT_NAME}',
                            exp_name=f'CBPR-switch-{args.mode}-{args.switch_human_freq}-learn', env_name='')
        else:
            logger = Logger(log_dir=f'./logs/cbpr/{LAYOUT_NAME}',
                            exp_name=f'CBPR-switch-{args.mode}-{args.switch_human_freq}-fix', env_name='')

        bpr_online.play(args, logger)







