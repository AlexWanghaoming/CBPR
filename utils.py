import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/PPO-discrete/')
import torch
import random
import cloudpickle
import pandas as pd
import numpy as np
import itertools
from typing import Callable
from overcooked_ai_py.mdp.overcooked_mdp import (OvercookedGridworld,)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--net_arch", type=str, default='mlp', help="policy net arch")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: 学习率线性衰减")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip: 0.1")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="value function coeffcient")
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args

def init_env(layout, lossless_state_encoding=False):
    mdp = OvercookedGridworld.from_layout_name(layout)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=600)
    if lossless_state_encoding:
        env = gym.make("Overcooked-v0", base_env=base_env, ego_featurize_fn=base_env.lossless_state_encoding_mdp, alt_featurize_fn=base_env.featurize_state_mdp)
    else:
        env = gym.make("Overcooked-v0", base_env=base_env, ego_featurize_fn=base_env.featurize_state_mdp, alt_featurize_fn=base_env.featurize_state_mdp )
    return env


class LinearAnnealer():  # reward shaping
    """Anneals a parameter from 1 to 0 over the course of training,
    over a specified horizon."""
    
    def __init__(self, horizon):
        self.horizon = horizon
    
    def param_value(self, timestep):
        if self.horizon == 0:
            return 0
        curr_value = max(1 - (timestep / self.horizon), 0)
        assert 0 <= curr_value <= 1
        return curr_value


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        if args.net_arch == "conv":
            self.s = np.zeros((args.batch_size, 5, 4, args.state_dim))
        else:
            self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        # self.s_ = np.zeros((args.batch_size, args.state_dim))
        if args.net_arch == "conv":
            self.s_ = np.zeros((args.batch_size, 5, 4, args.state_dim))
        else:
            self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done



class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
