import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/PPO-discrete/')
import random
from typing import *
import gym
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')
# print("当前系统路径", sys.path)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from torch.distributions import Categorical


def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Trick 5: policy entropy")
    parser.add_argument('--num_episodes',  type=int, default=3000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args


def init_env(layout, horizon=600, agent0_policy_name=None, agent1_policy_name=None, use_script_policy=False, old_dynamics=False):
    # mdp = OvercookedGridworld.from_layout_name(layout, start_state=OvercookedGridworld.get_random_start_state_fn)  # bug 随机游戏初始状态
    # wanghm Overcooked旧环境中，锅中放入三个菜后自动开始烹饪，所以无法烹饪原材料数量为2的菜品
    # if layout in ['counter_circuit', 'soup_coordination']:
    #     old_dynamic = False
    #     print(f'{layout} using old dynamic')

    marshmallow_experiment_shaped_rew = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
        "MIX_PUNISHMENT": 3
    }
    if layout in ['marshmallow_experiment']:
        mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=marshmallow_experiment_shaped_rew, old_dynamic=old_dynamics)
    else:
        mdp = OvercookedGridworld.from_layout_name(layout, old_dynamic=old_dynamics)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    env = gym.make("Overcooked-v0",
                   base_env=base_env,
                   agent0_policy_name=agent0_policy_name,
                   agent1_policy_name=agent1_policy_name,
                   use_script_policy=use_script_policy)
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
    def __init__(self, batch_size, state_dim, net_arch='mlp'):
        if net_arch == "conv":
            self.s = np.zeros((batch_size, 5, 4, state_dim))
        else:
            self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        # self.s_ = np.zeros((batch_size, state_dim))
        if net_arch == "conv":
            self.s_ = np.zeros((batch_size, 5, 4, state_dim))
        else:
            self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
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


# @ deprecated
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

# @ deprecated
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

# @ deprecated
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
        # x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)
    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def evaluate_actor(actor, s, deterministic=True, device='cpu'):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
    a_prob = actor(s)
    if deterministic:
        a = np.argmax(a_prob.detach().cpu().numpy().flatten())
    else:
        dist = Categorical(probs=a_prob)
        a = dist.sample().cpu().numpy()[0]
    return a


from scipy import stats

def print_mean_interval(data:list):
    # 计算平均值和标准误差
    mean = np.mean(data)
    std = np.std(data)
    sem = stats.sem(data)
    # 计算置信区间
    confidence = 0.95
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    print(f"\033[91m  mean: {mean}, 95% interval: {interval}, std: {std} \033[0m")

def limit_value(value, min=0.0001, max=0.9999):
    """限制值在0.01到0.99之间"""
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value