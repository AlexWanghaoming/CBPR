from copy import deepcopy
from collections import deque
import math
import numpy as np
import pickle
from typing import List, Tuple, Dict
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from bc.bc_hh import BehaviorClone
from models import MTP_MODELS, NN_MODELS, HP_MODELS, META_TASKS
from agents.ppo_discrete import PPO_discrete
from utils import seed_everything, init_env
import random
from state_trans_func.NN_scriptedPolicy import NN
import torch.nn as nn
import torch
from datetime import datetime
import wandb
from src.overcooked_ai_py.mdp.actions import Action


device = 'cpu'


class MetaTaskLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, tasks) -> Dict[str, str]:
        for idx, scripted_policy_name in enumerate(tasks):
            self.policy_lib[f'metatask_{idx+1}'] = scripted_policy_name
        return self.policy_lib


class AiPolicyLibrary:
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


def eval_rollouts(env, ai_agent: PPO_discrete):
    """
    在生成performance model阶段作评估
    """
    obs = env.reset()
    ai_obs, h_obs = obs['both_agent_obs']
    accumulated_reward = 0
    done = False
    while not done:
        # 策略库中的agent和人玩，得到rewards
        # ai_act = agent_policy.predict(ai_obs)[0]
        # ai_act = evaluate(agent_policy, ai_obs)  # 确定性策略
        ai_act = ai_agent.evaluate(ai_obs)  # 随机策略
        obs, sparse_reward, done, info = env.step((ai_act, 1))
        ai_obs, h_obs = obs['both_agent_obs']
        accumulated_reward += sparse_reward
    return accumulated_reward


class BPR_offline:
    """
    BPR离线部分，performance model初始化，belief初始化
    """
    def __init__(self, args):
        self.human_policys = HPL.gen_policy_library(tasks=META_TASKS[args.layout])
        self.ai_policys = APL.gen_policy_library(args)


    def gen_performance_model(self) -> List[Dict[str, Dict[str, float]]]:
        performance_model_save_path = f'../models/performance/init_performance_{LAYOUT_NAME}.pkl'
        if os.path.exists(performance_model_save_path):
            with open(performance_model_save_path, 'rb') as ff:
                performance_model = pickle.load(ff)
            return performance_model

        performance_model = [{}, {}]
        n_rounds = 50
        for h_ in self.human_policys: # h_: "metatask1"
            env = init_env(layout=args.layout,
                           agent0_policy_name='ppo',
                           agent1_policy_name=f'script:{self.human_policys[h_]}',
                           use_script_policy=True)
            temp_mean = {}
            temp_std = {}
            for ai_ in self.ai_policys:
                ai_policy = self.ai_policys[ai_]
                u_list = []
                for _ in range(n_rounds):
                    episodic_reward = eval_rollouts(env, ai_policy)
                    u_list.append(episodic_reward)
                temp_mean[ai_] = np.mean(u_list)
                temp_std[ai_] = np.std(u_list)
            performance_model[0][h_] = temp_mean
            performance_model[1][h_] = temp_std
        with open(performance_model_save_path, 'wb') as f:
            pickle.dump(performance_model, f)
        return performance_model

    def gen_belief(self) -> Dict[str, float]:
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
                 human_policys: Dict[str, str],
                 performance_model: List[Dict[str, Dict[str, float]]],
                 NN_models: Dict[str, nn.Module],
                 belief: Dict[str, float],
                 new_polcy_threshold=0.3):
        self.performance_model = performance_model
        self.belief = belief
        self.agents = agents
        self.human_policys = human_policys
        self.NNs = NN_models
        self.threshold = new_polcy_threshold
        self.eps = 5e-6

    def play(self, args):
        human_model = torch.load(HP_MODELS[args.layout], map_location=device)
        args.max_episode_steps = 600
        total_steps = 0
        env = init_env(layout=args.layout)
        for k in range(args.num_episodes):
            Q = deque(maxlen=args.Q_len)  # 记录maxlen条人类交互数据 (s,a,s',r)
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
                ai_act = best_agent.evaluate(ai_obs)
                h_act = human_model.choose_action(h_obs, deterministic=False)
                obs_, sparse_reward, done, info = env.step((ai_act, h_act))
                ep_reward += sparse_reward
                ai_obs_, h_obs_ = obs_['both_agent_obs']

                ###############   在原始的OKR上做以下改动   ##################
                # h_dire = info['joint_action'][1]
                # h_act = Action.INDEX_TO_ACTION.index(h_dire)
                actions_one_hot = np.eye(6)[h_act]
                obs_x = np.hstack([h_obs, actions_one_hot])
                obs_x = torch.from_numpy(obs_x).float().to(device)
                obs_y = np.hstack([h_obs_, sparse_reward])
                obs_y = torch.from_numpy(obs_y).float().to(device)
                Q.append((obs_x, obs_y))  # wanghm 将当前时间步的人类（s,a,s',r）加入经验池
                ai_obs, h_obs = ai_obs_, h_obs_
                self.xi = self._update_xi(self.human_policys, Q)  # 更新intra-episode belief \xi
                # print('xi: ', self.xi)
                ###############   在原始的OKR上做以上改动   ##################

                self.zeta = self._update_zeta(t=episode_steps, rho=args.rho)  # 更新integrated belief \zeta
                # print('zeta: ', self.zeta)
                ### debug: 直接选对应的策略作为最优策略
                # best_agent_id = list(self.agents.keys())[policy_idx-1]
                # best_agent = self.agents[best_agent_id]

                best_agent_id, best_agent = self._reuse_optimal_policy(inter_episode=False)  # 轮内重用最优策略

                ### log
                # if best_agent_id != best_agent_id_prime:
                #     # print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                #     best_agent_id_prime = best_agent_id
                self.xi = deepcopy(self.zeta)  # 每一步更新 \xi
            print(f'Ep {k + 1} rewards: {ep_reward}')
            # wandb.log({'episode': k+1, 'ep_reward': ep_reward})
            # 更新本轮的belief
            self.belief = deepcopy(self.xi)
            self.belief = self._update_beta(best_agent_id, ep_reward)

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

    def _update_xi(self, human_policy, Q) -> Dict[str, float]:
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
                NN = self.NNs[id]
                for q in Q:
                    s_a, s_prime_r = q
                    means = NN(s_a)
                    vars = torch.tensor([0.1]).to(device)
                    std_devs = torch.sqrt(vars)
                    probs = torch.exp(-0.5 * ((s_prime_r - means) / std_devs) ** 2) / (
                                std_devs * torch.sqrt(2 * torch.tensor([3.141592653589793]).to(device)))
                    probs_mean = torch.mean(probs).item()
                    su += np.log(probs_mean)
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
        target_task = max(belief, key=belief.get) # "metatask_1"
        idx = target_task.split('_')[-1]
        best_agent_name = 'mtp_' + idx    # "mtp_1"
        best_agent = self.agents[best_agent_name]
        return best_agent_name, best_agent

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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--Q_len', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.5, help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    LAYOUT_NAME = args.layout
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    # wandb.init(project='overcooked_rl',
    #            group='BPR',
    #            name=f'bprOKR_{args.layout}_HP_seed{args.seed}',
    #            config=vars(args),
    #            job_type='eval',
    #            reinit=True)
    wandb.init
    seed_everything(args.seed)
    HPL = MetaTaskLibrary()
    h_pl = HPL.gen_policy_library(tasks=META_TASKS[args.layout])  # 构建Human策略库
    NNL = NNLibrary()
    NN_models = NNL.gen_policy_library()
    APL = AiPolicyLibrary()
    apl = APL.gen_policy_library(args)  # 构建AI策略库
    bpr_offline = BPR_offline(args)
    performance_model = bpr_offline.gen_performance_model()
    print("初始performance_model: ", performance_model)
    belief = bpr_offline.gen_belief()
    bpr_online = BPR_online(agents=APL.gen_policy_library(args),
                            human_policys=HPL.gen_policy_library(tasks=META_TASKS[args.layout]),
                            NN_models=NN_models,
                            performance_model=performance_model,
                            belief=belief)
    bpr_online.play(args)
    # wandb.finish()







