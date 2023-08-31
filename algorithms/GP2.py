import pickle
import numpy as np
import GPy
import torch
from gpytorch.models import ExactGP
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gptorch.likelihoods import Gaussian
from gptorch.mean_functions import Zero
from gptorch.kernels import Rbf

from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS, BCP_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from utils import *
from typing import *


class GPStateTransitionModel:
    def __init__(self, action_dim=6, state_dim=96):
        self.model_s_prime = None
        self.model_r = None
        self.action_dim = action_dim
        self.state_dim = state_dim

    def train(self, states, actions, s_prime, rewards, epochs=100, lr=0.1):
        """
        使用给定的数据训练GP模型。
        
        参数:
        - states: 状态数据，形状为[num_samples, state_dim]
        - actions: 动作数据，形状为[num_samples]
        - s_prime_r: 输出数据，形状为[num_samples, state_dim + 1] (最后一列为奖励)
        """
        # 将动作转换为one-hot编码
        actions_one_hot = torch.eye(self.action_dim)[actions]

        # 拼接状态和动作
        s_a = torch.cat([states, actions_one_hot], dim=1)

        likelihood_s_prime = Gaussian()
        kernel_s_prime = Rbf(input_dim=s_a.shape[1])
        self.model_s_prime = GPR(s_a, s_prime, kernel=kernel_s_prime, likelihood=likelihood_s_prime, mean_function=Zero)

        likelihood_r = Gaussian()
        kernel_r = Rbf(input_dim=s_a.shape[1])
        self.model_r = GPR(s_a, rewards, kernel=kernel_r, likelihood=likelihood_r, mean_function=Zero)

        optimizer = torch.optim.Adam(self.model_s_prime.parameters(), lr = lr)
        self.model_s_prime.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = -self.model_s_prime.marginal_log_likelihood()
            loss.backword()
            optimizer.step()

        optimizer = torch.optim.Adam(self.model_r.parameters(), lr = lr)
        self.model_r.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            loss = -self.model_r.marginal_log_likelihood()
            loss.backword()
            optimizer.step()

    def predict(self, states, actions):
        """
        使用训练好的GP模型预测s'和r。
        
        参数:
        - states: 状态数据，形状为[num_samples, state_dim]
        - actions: 动作数据，形状为[num_samples]
        
        返回:
        - 预测的s'和r
        """
        actions_one_hot = torch.eye(self.action_dim)[actions]
        s_a = torch.cat([states, actions_one_hot], dim=1)
        self.model_s_prime.eval()
        self.model_r.eval()
        with torch.no_grad():
            s_prime_pred, _ = self.model_s_prime(s_a)
            r_pred, _ = self.model_r(s_a)
        return s_prime_pred, r_pred

    def save_gp(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.model_s_prime, self.model_r), file)

    def load_gp(self, filename):
        with open(filename, 'rb') as file:
            self.model_s_prime, self.model_r = pickle.load(file)


def collect_trajs(args, ego_agent:PPO_discrete, alt_agent:nn.Module) -> Dict[Tuple,Dict[str, np.ndarray]]:
    N = 50000   # 每个 meta-task需要收集 N 条数据用于训练 GP
    meta_task_trajs = {}
    meta_task_steps = {}
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    for k in range(1, args.num_episodes + 1):
        obs = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        done = False
        episode_reward = 0
        while not done:
            key = tuple(alt_obs[slice(4, 8)])
            if key not in meta_task_trajs:
                meta_task_trajs[key] = {'train_s':[], 'train_a':[], 'train_r':[], 'train_s_':[]}
                meta_task_steps[key] = 0
            meta_task_steps[key] += 1

            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)  # Action and the corresponding log probability
            alt_a = alt_agent.choose_action(alt_obs, deterministic=True)  # Action and the corresponding log probability

            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += sparse_reward

            if meta_task_steps[key] <= N:
                meta_task_trajs[key]['train_s'].append(alt_obs)
                meta_task_trajs[key]['train_a'].append(alt_a)
                meta_task_trajs[key]['train_r'].append(sparse_reward)
                meta_task_trajs[key]['train_s_'].append(alt_obs_)

            ego_obs = ego_obs_
            alt_obs = alt_obs_

        # print(f"Ep {k} test reward:", episode_reward)

        # 当所有meta-task的数据收集齐之后，退出循环
        if k > 15000:
            if all(value >= N for value in meta_task_steps.values()):
                break
    for key in meta_task_trajs:
        for i in meta_task_trajs[key]:
            meta_task_trajs[key][i] = np.array(meta_task_trajs[key][i])

    return meta_task_trajs


if __name__ == "__main__":
    ### debug
    states_train = torch.rand(100, 96)
    actions_train = torch.randint(0, 6, (100,))
    s_prime_train = torch.rand(100, 96)
    rewards_train = torch.rand(100, 1)
    model = GPStateTransitionModel()
    model.train(states_train, actions_train, s_prime_train, rewards_train)
    states_test = torch.rand(5, 96)
    actions_test = torch.randint(0, 6, (5,))
    s_prime_pred, r_pred = model.predict(states_test, actions_test)
    print("Predicted s':", s_prime_pred)
    print("Predicted r:", r_pred)

    LOAD_DATA = False
    default_args = parse_args()
    args = default_args
    args.layout = "cramped_room"
    # args.layout = 'marshmallow_experiment'
    if LOAD_DATA:
        args.num_episodes = 20000
        test_env = init_env(layout=args.layout, lossless_state_encoding=False)
        args.state_dim = test_env.observation_space.shape[0]
        args.action_dim = test_env.action_space.n

        ai = PPO_discrete(args)
        ai.load_actor(BCP_MODELS[args.layout])
        h = torch.load(BC_MODELS[args.layout], map_location='cpu')

        meta_task_trajs = collect_trajs(args, ego_agent=ai, alt_agent=h)
        f_save = open('gp_trajs.pkl', 'wb')
        pickle.dump(meta_task_trajs, f_save)
        f_save.close()
    else:
        f_read = open('gp_trajs.pkl', 'rb')
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






