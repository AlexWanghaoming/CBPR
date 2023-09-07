import time
from copy import deepcopy
import numpy as np
import re
from typing import Tuple, Dict
import argparse
import torch
import torch.nn as nn
import os, sys
from models import MTP_MODELS, META_TASK_MODELS, GP_MODELS, NN_MODELS
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from agents.ppo_discrete import PPO_discrete
from utils import seed_everything, init_env
from rl_plotter.logger import Logger
import random
from state_trans_func.NN import NN

device = 'cuda'

class MetaTaskLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self) -> Dict[str, nn.Module]:
        for bc_model_path in META_TASK_MODELS[LAYOUT_NAME]:
            pattern = r'\(\d+\.\d+(?:, \d+\.\d+)*\)'
            user_name = re.findall(pattern, bc_model_path)[0]
            policy = torch.load(bc_model_path)
            self.policy_lib[user_name] = policy
        return self.policy_lib


class MTPLibrary:
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


class NNLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self) -> Dict[str, nn.Module]:
        for bc_model_path in NN_MODELS[LAYOUT_NAME]:
            pattern = r'\(\d+\.\d+(?:, \d+\.\d+)*\)'
            user_name = re.findall(pattern, bc_model_path)[0]
            state_dict = torch.load(bc_model_path)
            model = NN(input_dim=102, output_dim=97)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self.policy_lib[user_name] = model
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
                 NN_models: Dict[str, nn.Module],
                 belief: Dict[str, float],
                 new_polcy_threshold=0.3):

        self.belief = belief
        self.mtp = agents
        self.meta_tasks = human_policys
        self.NNs = NN_models
        self.threshold = new_polcy_threshold
        self.eps = 5e-7

        self.env = init_env(layout=LAYOUT_NAME, lossless_state_encoding=False)

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

            episode_steps = 0

            best_agent_id_prime = ""
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            best_agent_id, best_agent = self._reuse_optimal_policy()

            while not done:
                # best_agent_id, best_agent = self._reuse_optimal_policy()

                total_steps += 1
                episode_steps += 1
                if args.mode == "intra":
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

                # wanghm 用NN作预测
                actions_one_hot = np.eye(6)[h_act]
                obs_x = np.hstack([h_obs, actions_one_hot])
                obs_x = torch.from_numpy(obs_x).float().to(device)
                obs_y = np.hstack([h_obs_, sparse_reward]) # s_prime, r
                # obs_y = h_obs_ # s_prime
                obs_y = torch.from_numpy(obs_y).float().to(device)

                self.belief = self._update_beta(obs_x=obs_x,
                                                obs_y=obs_y)
                ai_obs = ai_obs_
                h_obs = h_obs_

                # ### debug: 直接选对应的策略作为最优策略
                best_agent_id = list(self.mtp.keys())[policy_idx-1]
                best_agent = self.mtp[best_agent_id]

                # debug
                if best_agent_id != best_agent_id_prime:
                    print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                    best_agent_id_prime = best_agent_id

                # env.render()

            print(f'Ep {k + 1} rewards: {ep_reward}')
            logger.update(score=[ep_reward], total_steps=k + 1)
            # print("----------------------------------------------------------------------------------------------------")
            # 更新本轮的belief


        # return rewards_list

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
            new_belief[id] = (p_temp[id] + eps) / (sum(p_temp.values()) )

        # print(new_belief)
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

        best_agent_id, best_agent = max(self.belief, key=self.belief.get), \
            self.mtp[max(self.belief, key=self.belief.get)]
        return best_agent_id, best_agent

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

    parser.add_argument('--device', type=str, default='cuda')
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
        HPL = MetaTaskLibrary()
        meta_task_lib = HPL.gen_policy_library()  # 构建Human策略库
        APL = MTPLibrary()
        mtp_lib = APL.gen_policy_library(args)  # 构建AI策略库
        NNL = NNLibrary()
        NN_models = NNL.gen_policy_library()

        bpr_offline = BPR_offline(args)
        belief = bpr_offline.gen_belief()
        bpr_online = BPR_online(agents=mtp_lib,
                                human_policys=meta_task_lib,
                                NN_models=NN_models,
                                belief=belief,
                                new_polcy_threshold=0.3)

        if args.new_policy_learning:
            logger = Logger(log_dir=f'./logs/cbpr/{LAYOUT_NAME}',
                            exp_name=f'CBPR-switch-{args.mode}-{args.switch_human_freq}-learn', env_name='')
        else:
            logger = Logger(log_dir=f'./logs/cbpr/{LAYOUT_NAME}',
                            exp_name=f'CBPR-switch-{args.mode}-{args.switch_human_freq}-fix-gold', env_name='')

        bpr_online.play(args, logger)







