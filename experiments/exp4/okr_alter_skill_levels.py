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
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import MTP_MODELS, METATASK_MODELS, META_TASKS, SKILL_MODELS
from agents.ppo_discrete import PPO_discrete
from My_utils import seed_everything, init_env, limit_value, evaluate_actor
import random
from bc.opponent_scriptedPolicy import Opponent
from bc.bc_hh import BehaviorClone
import torch.nn as nn
import torch
import wandb
from src.overcooked_ai_py.mdp.actions import Action

device = 'cuda'

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
        for idx, mtp_path in enumerate(MTP_MODELS[args.layout]):
            agent_id = f'mtp_{idx+1}'
            agent = PPO_discrete()
            agent.load_actor(mtp_path)
            # agent.load_critic(mtp_path[1])
            self.policy_lib[agent_id] = agent
        return self.policy_lib


class MTLibrary:
    def __init__(self):
        self.policy_lib = {}

    def gen_policy_library(self, args) -> Dict[str, nn.Module]:
        for idx, mt_model in enumerate(METATASK_MODELS[args.layout]):
            state_dict = torch.load(mt_model)
            # model = Opponent(state_dim=96, hidden_dim=128, action_dim=6)
            model = BehaviorClone(state_dim=96, hidden_dim=128, action_dim=6)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self.policy_lib[f'metatask_{idx+1}'] = model
        return self.policy_lib


def eval_rollouts(env, ai_agent: PPO_discrete):
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
    def __init__(self, args, HPL, APL):
        self.human_policys = HPL.gen_policy_library(tasks=META_TASKS[args.layout])
        self.ai_policys = APL.gen_policy_library(args)
        self.args = args


    def gen_performance_model(self) -> List[Dict[str, Dict[str, float]]]:
        performance_model_save_path = f'/alpha/overcooked_rl/models/performance/init_performance_{self.args.layout}.pkl'
        if os.path.exists(performance_model_save_path):
            with open(performance_model_save_path, 'rb') as ff:
                performance_model = pickle.load(ff)
            return performance_model

        performance_model = [{}, {}]
        n_rounds = 50
        for h_ in self.human_policys: # h_: "metatask1"
            env = init_env(layout=self.args.layout,
                           agent0_policy_name='mtp',
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
                 MT_models: Dict[str, nn.Module],
                 belief: Dict[str, float]):
        self.performance_model = performance_model
        self.belief = belief
        self.agents = agents
        self.human_policys = human_policys
        self.MTs = MT_models
        self.eps = 1e-7

    def play(self, args):
        args.max_episode_steps = 600
        total_steps = 0
        # 初始化人类模型和策略
        policy_idx = 2
        print(f"初始策略 metatask_{policy_idx}: ", META_TASKS[args.layout][policy_idx - 1])
        env = init_env(layout=args.layout, use_script_policy=False)
        skill_model_path = SKILL_MODELS[args.layout][policy_idx - 1]
        skill_model = torch.load(skill_model_path, map_location=device)
        for k in range(args.num_episodes):
            if args.mode == 'inter':
                policy_idx = policy_id_list.pop()
                skill_model_path = SKILL_MODELS[args.layout][policy_idx-1]
                skill_model = torch.load(skill_model_path, map_location=device)
                # print(f"人类改变至策略 metatask_{policy_idx}: ", META_TASKS[args.layout][policy_idx - 1])  # log
            Q = deque(maxlen=args.Q_len)
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
                if args.mode == "intra":
                    # 轮内切换人的策略
                    if episode_steps % args.switch_human_freq == 0:
                        policy_idx = policy_id_list.pop()
                        # print(f"人类改变至策略 metatask_{policy_idx}: ", META_TASKS[args.layout][policy_idx - 1])  # log
                        skill_model_path = SKILL_MODELS[args.layout][policy_idx - 1]
                        skill_model = torch.load(skill_model_path, map_location=device)
                ai_act = best_agent.evaluate(ai_obs)
                h_act = evaluate_actor(skill_model, h_obs, deterministic=False, device=device)
                obs_, sparse_reward, done, info = env.step((ai_act, h_act))
                ep_reward += sparse_reward
                ai_obs, h_obs = obs_['both_agent_obs']
                h_dire = info['joint_action'][1]
                h_obs = torch.tensor(h_obs, dtype=torch.float32).to(device)
                h_act = torch.tensor(np.array([h_act]), dtype=torch.int64).to(device)
                Q.append((h_obs, h_act))

                # if episode_steps % 1 == 0 and len(Q) == args.Q_len:
                self.xi = self._update_xi(Q)  # 更新intra-episode belief $\xi$ 原文公式8,9
                # print('xi: ', self.xi)
                self.zeta = self._update_zeta(t=episode_steps, rho=args.rho)  # 更新integrated belief $\zeta$ 原文公式10
                best_agent_id, best_agent = self._reuse_optimal_policy(belief=self.zeta)  # 轮内重用最优策略
                self.xi = deepcopy(self.zeta)  # 每一步更新 \xi

                # ### debug: 直接选对应的策略作为最优策略
                # 240 240 320 300 260 280 220 220 320 160
                # best_agent_id = list(self.agents.keys())[policy_idx-1]
                # best_agent = self.agents[best_agent_id]
                # if best_agent_id != best_agent_id_prime:
                #     print(f'CBPR重用策略 {best_agent_id} 和人合作!')
                #     best_agent_id_prime = best_agent_id
            # print(f'Ep {k + 1} rewards: {ep_reward}')
            wandb.log({'episode': k+1, 'ep_reward': ep_reward})

            # 更新本轮的belief
            self.belief = deepcopy(self.xi)
            self.belief = self._update_beta(best_agent_id, ep_reward)

    def _update_zeta(self, t, rho):
        new_zeta = {}
        for id in self.belief:
            new_zeta[id] = (rho ** t) * self.belief[id] + (1 - rho ** t) * self.xi[id]
        return new_zeta

    def _update_beta(self, agent_id: str, u: float) -> Dict[str, float]:
        """
        每 episode 利用Bayesian公式更新回合间belif
        belief: dict
        """
        p_temp = {}
        new_belief = {}
        for id in self.belief:
            p_temp[id] = self._gen_pdf(self.performance_model[0][id][agent_id],
                                       self.performance_model[1][id][agent_id],
                                       u) * self.belief[id]
        for id in self.belief:
            new_belief[id] = (p_temp[id] + self.eps) / (sum(p_temp.values()) + self.eps * 4)

        new_belief = {key: limit_value(value) for key, value in new_belief.items()}
        return new_belief

    def _update_xi(self, Q) -> Dict[str, float]:
        """
        每 step 利用Bayesian公式更新回合内belif
        xi: dict
        """
        def gen_Q_prob(Q):
            Q_prob = {}
            temp = {}
            for id in self.xi:
                su = 0
                opponent_model = self.MTs[id]
                for q in Q:
                    s, a = q
                    # su += torch.log(opponent_model(s.unsqueeze(dim=0), a.unsqueeze(dim=1))).item()   # opponent model
                    su += np.log(opponent_model.action_probability(s)[a])   # behavior cloning model
                temp[id] = np.exp(su)

            for id in self.xi:
                Q_prob[id] = (temp[id] + self.eps) / (sum(temp.values()) + self.eps * 4)
            return Q_prob
        p_temp = {}
        new_xi = {}
        Q_prob = gen_Q_prob(Q)
        for id in self.xi:
            p_temp[id] = Q_prob[id] * self.xi[id]
        for id in self.xi:
            new_xi[id] = (p_temp[id] + self.eps) / (sum(p_temp.values()) + self.eps * 4)
        new_xi = {key: limit_value(value) for key, value in new_xi.items()}
        return new_xi

    def _gen_pdf(self, mu, sigma, x):
        sigma = sigma + self.eps
        return (1 / (sigma * (math.pi * 2) ** 0.5)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def _reuse_optimal_policy(self, belief) -> Tuple[str, PPO_discrete]:
        """
        calculate expectation of utility
        用BPR-EI求最优策略
        return: 最优策略id，最优策略
        """
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
        target_task = max(belief, key=belief.get)    # metatask_1
        idx = target_task.split('_')[-1]
        best_agent_name = 'mtp_' + idx      # mtp_1
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
    parser.add_argument('--layout', default='soup_coordination')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--mode', default='intra', help='swith policy inter or intra')
    parser.add_argument("--switch_human_freq", type=int, default=100, help="Frequency of switching human policy")
    parser.add_argument('--Q_len', type=int, default=5)
    parser.add_argument('--rho', type=float, default=0.1, help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
    args = parse_args()
    LAYOUT_NAME = args.layout
    wandb.init(project='overcooked_rl',
               group='exp4',
               name=f'okr_{args.layout}_{args.mode}{args.switch_human_freq}_seed{args.seed}_Q{args.Q_len}',
               config=vars(args),
               job_type='eval',
               dir=os.path.join(WANDB_DIR, 'exp4', 'okr'),
               reinit=True)

    if args.mode == 'inter':
        random.seed(42)
        N = args.num_episodes // args.switch_human_freq
        policy_id_list = [random.randint(1, 3) for _ in range(N)]  # 固定测试的策略顺序
        policy_id_list = [val for val in policy_id_list for i in range(args.switch_human_freq)]
    if args.mode == 'intra':
        random.seed(42)
        N = args.num_episodes * (600 // args.switch_human_freq)
        policy_id_list = [random.randint(1, 3) for _ in range(N)]

    seed_everything(args.seed)
    HPL = MetaTaskLibrary()
    h_pl = HPL.gen_policy_library(tasks=META_TASKS[args.layout])  # 构建Human策略库
    MTL = MTLibrary()
    MT_models = MTL.gen_policy_library(args)
    APL = AiPolicyLibrary()
    apl = APL.gen_policy_library(args)  # 构建AI策略库
    bpr_offline = BPR_offline(args, HPL=HPL, APL=APL)
    performance_model = bpr_offline.gen_performance_model()
    print("初始performance_model: ", performance_model)
    belief = bpr_offline.gen_belief()
    bpr_online = BPR_online(agents=APL.gen_policy_library(args),
                            human_policys=HPL.gen_policy_library(tasks=META_TASKS[args.layout]),
                            MT_models=MT_models,
                            performance_model=performance_model,
                            belief=belief)
    bpr_online.play(args)
    wandb.finish()







