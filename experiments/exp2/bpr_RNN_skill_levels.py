from copy import deepcopy
import numpy as np
from typing import *
import argparse
import torch
import torch.nn as nn
import os, sys
from collections import deque
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import MTP_MODELS, NN_MODELS, SKILL_MODELS, META_TASKS
from agents.ppo_discrete import PPO_discrete
from state_trans_func.RNN_scriptedPolicy import RNNPredictor
from My_utils import seed_everything, init_env, evaluate_actor, print_mean_interval
import math
from src.overcooked_ai_py.mdp.actions import Action
import wandb


device = 'cuda'
INPUT_LENGTH = 30
PREDICT_LENGTH = 10


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
        for idx, nn_model_path in enumerate(NN_MODELS[LAYOUT_NAME]):
            state_dict = torch.load(nn_model_path)
            model = RNNPredictor(input_size=102, hidden_size=128, num_layers=1, output_size=96 * 10)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self.policy_lib[f'metatask_{idx+1}'] = model
        return self.policy_lib


class BPR_offline:

    def __init__(self, args):
        HPL = MetaTaskLibrary()
        self.human_policys = HPL.gen_policy_library(tasks=META_TASKS[args.layout])
        self.ai_policys = APL.gen_policy_library(args)

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
        # print('Initial belief is ', beta)
        ss = deepcopy(beta)
        return ss


class BPR_online:
    def __init__(self, agents: Dict[str, PPO_discrete],
                 human_policys: Optional[Dict[str, nn.Module]],
                 NN_models: Dict[str, nn.Module],
                 belief: Dict[str, float]):
        self.belief = belief
        self.mtp = agents
        self.meta_tasks = human_policys
        self.NNs = NN_models
        self.eps = 1e-6

    def play(self, args, skill_model):
        args.max_episode_steps = 600
        total_steps = 0
        env = init_env(layout=args.layout)
        r_list = []
        for k in range(args.num_episodes):
            Q = deque(maxlen=args.Q_len)
            episode_steps = 0
            obs = env.reset()
            ai_obs, h_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            while not done:
                best_agent_id, best_agent = self._reuse_optimal_policy()  # 选择belief最大的智能体
                total_steps += 1
                episode_steps += 1
                ai_act = best_agent.evaluate(ai_obs)  # 智能体选动作
                h_act = evaluate_actor(skill_model, h_obs, deterministic=False)
                obs_, sparse_reward, done, info = env.step((ai_act, h_act))
                ep_reward += sparse_reward
                ai_obs_, h_obs_ = obs_['both_agent_obs']
                h_dire = info['joint_action'][1]
                h_act = Action.INDEX_TO_ACTION.index(h_dire)
                actions_one_hot = np.eye(6)[h_act]
                Q.append(np.hstack([h_obs, actions_one_hot]))
                if len(Q) == args.Q_len:
                    self.belief = self._update_beta(Q)
                ai_obs, h_obs = ai_obs_, h_obs_

            print(f'Ep {k + 1} rewards: {ep_reward}')
            r_list.append(ep_reward)
            # wandb.log({'episode': k+1, 'ep_reward': ep_reward})
        print_mean_interval(r_list)

    def _update_beta(self,Q) -> Dict[str, float]:
        """
        这里和BPR NN *Efficient Bayesian Policy Reuse With a Scalable Observation Model in Deep Reinforcement Learning, TNNLS* 有区别
        belief: dict
        """
        state_seq = torch.tensor(Q, dtype=torch.float32).to(device)
        input_seqs = state_seq[:INPUT_LENGTH].unsqueeze(dim=0)
        target_seqs = state_seq[INPUT_LENGTH:][:, :96].unsqueeze(dim=0)

        var = torch.tensor([0.01]).to(device)
        std_dev = torch.sqrt(var)
        temp = {}
        new_belief = {}
        for mt in self.belief:
            NN = self.NNs[mt]
            means = NN(input_seqs)  # (1, 960)  # 输入30个时间步的s,a，NN预测10个时间步（10*96）的状态，
            probs = (1 / (std_dev * (math.pi * 2) ** 0.5)) * torch.exp(-(means - target_seqs.contiguous().view(target_seqs.size(0), -1)) ** 2 / (2 * std_dev ** 2))
            su = torch.mean(probs).item()
            # print(mt, su)
            temp[mt] = su * self.belief[mt]

        for mt in self.belief:
            new_belief[mt] = (temp[mt] + self.eps) / (sum(temp.values()) + self.eps*len(META_TASKS[args.layout]))

        # print('new belief:', new_belief)
        return new_belief


    def _reuse_optimal_policy(self) -> Tuple[str, PPO_discrete]:
        target_task = max(self.belief, key=self.belief.get) # "metatask_1"
        idx = target_task.split('_')[-1]
        best_agent_name = 'mtp_' + idx # "mtp_1"
        best_agent = self.mtp[best_agent_name]
        return best_agent_name, best_agent



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Bayesian policy reuse algorithm on overcooked''')
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Q_len', type=int, default=40)
    parser.add_argument('--skill_level', default='low', help='low or medium or high')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
    args = parse_args()
    LAYOUT_NAME = args.layout
    if args.skill_level == 'low':
        skill_model_path = SKILL_MODELS[args.layout][0]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    elif args.skill_level == 'medium':
        skill_model_path = SKILL_MODELS[args.layout][1]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    elif args.skill_level == 'high':
        skill_model_path = SKILL_MODELS[args.layout][2]
        skill_model = torch.load(skill_model_path, map_location='cpu')
    else:
        pass
    # wandb.init(project='overcooked_rl',
    #            group='exp2',
    #            name=f'bprRNN_{args.layout}_{args.skill_level}_seed{args.seed}',
    #            config=vars(args),
    #            job_type='eval',
    #            dir=os.path.join(WANDB_DIR, 'exp2'),
    #            reinit=True)

    seed_everything(args.seed)
    APL = MTPLibrary()
    mtp_lib = APL.gen_policy_library(args)  # 构建AI策略库
    NNL = NNLibrary()
    NN_models = NNL.gen_policy_library()
    bpr_offline = BPR_offline(args)
    belief = bpr_offline.gen_belief()
    bpr_online = BPR_online(agents=mtp_lib,
                            human_policys=None,
                            NN_models=NN_models,
                            belief=belief)

    bpr_online.play(args, skill_model)
    # wandb.finish()






