"""
@ deprecated

This GPy package can not be use, because out-of-memory errors over large datesets
"""

import pickle
import numpy as np
import GPy
import torch
from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS, BCP_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from My_utils import *
from typing import *


class GPStateTransitionModel:
    def __init__(self, action_dim=6, state_dim=96):
        self.model_s_prime = None
        self.model_r = None
        self.action_dim = action_dim
        self.state_dim = state_dim

    def train(self, states, actions, s_prime, rewards):
        """
        使用给定的数据训练GP模型。
        
        参数:
        - states: 状态数据，形状为[num_samples, state_dim]
        - actions: 动作数据，形状为[num_samples]
        - s_prime_r: 输出数据，形状为[num_samples, state_dim + 1] (最后一列为奖励)
        """
        # 将动作转换为one-hot编码
        actions_one_hot = np.eye(self.action_dim)[actions]

        # 拼接状态和动作
        s_a = np.hstack([states, actions_one_hot])

        # 分离s'和r
        # s_prime = s_prime_r[:, :-1]
        # r = s_prime_r[:, -1].reshape(-1, 1)

        # 为s'和r分别拟合GP模型
        self.model_s_prime = GPy.models.GPRegression(s_a, s_prime)
        self.model_r = GPy.models.GPRegression(s_a, rewards)

        # 优化模型
        self.model_s_prime.optimize()
        self.model_r.optimize()

    def predict(self, states, actions):
        """
        使用训练好的GP模型预测s'和r。
        
        参数:
        - states: 状态数据，形状为[num_samples, state_dim]
        - actions: 动作数据，形状为[num_samples]
        
        返回:
        - 预测的s'和r
        """
        actions_one_hot = np.eye(self.action_dim)[actions]
        s_a = np.hstack([states, actions_one_hot])

        s_prime_pred, _ = self.model_s_prime.predict(s_a)
        r_pred, _ = self.model_r.predict(s_a)
        return s_prime_pred, r_pred

    def save_gp(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.model_s_prime, self.model_r), file)

    def load_gp(self, filename):
        with open(filename, 'rb') as file:
            self.model_s_prime, self.model_r = pickle.load(file)


if __name__ == "__main__":
    ### debug
    # states_train = np.random.rand(100, 96)
    # actions_train = np.random.randint(0, 6, 100)
    # s_prime_train = np.random.rand(100, 96)
    # rewards_train = np.random.rand(100, 1)
    # model = GPStateTransitionModel()
    # model.train(states_train, actions_train, s_prime_train, rewards_train)
    # states_test = np.random.rand(5, 96)
    # actions_test = np.random.randint(0, 6, 5)
    # s_prime_pred, r_pred = model.predict(states_test, actions_test)
    # print("Predicted s':", s_prime_pred)
    # print("Predicted r:", r_pred)

    default_args = parse_args()
    args = default_args
    # args.layout = "cramped_room"
    args.layout = 'marshmallow_experiment'

    f_read = open(f'gp_trajs_{args.layout}.pkl', 'rb')
    meta_task_trajs = pickle.load(f_read)
    f_read.close()
    for key in meta_task_trajs:
        states_train = meta_task_trajs[key]['train_s'][:10000]
        actions_train = meta_task_trajs[key]['train_a'][:10000]
        rewards_train = np.reshape(meta_task_trajs[key]['train_r'], (-1,1))[:10000]
        s_prime_train = meta_task_trajs[key]['train_s_'][:10000]

        model = GPStateTransitionModel()
        model.train(states_train, actions_train, s_prime_train, rewards_train)
        model.save_gp(f'../models/gp/gp_model_{key}_{args.layout}.pkl')






