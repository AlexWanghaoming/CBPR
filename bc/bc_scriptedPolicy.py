import torch
import torch.nn as nn
import math
import os, sys
import numpy as np
import argparse
from My_utils import init_env
from models import MTP_MODELS, META_TASKS, BCP_MODELS, SP_MODELS
from bc_hh import BehaviorClone, train
from src.overcooked_ai_py.mdp.actions import Action
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda'


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--layout', type=str, default='coordination_ring')
    parser.add_argument('--layout', type=str, default='asymmetric_advantages')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    opt = parser.parse_args()
    return opt


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


if __name__ == '__main__':
    args = parse_opt()
    data_collection_epochs = 100
    warmup_epochs = 2
    initial_lr = 0.001
    for idx, meta_task in enumerate(META_TASKS[args.layout]):
        # agent_path = BCP_MODELS[LAYOUT_NAME]  # bcp agent的能力最差， 和script policy agent 合作可以见到尽可能大的状态空间
        agent_path = MTP_MODELS[args.layout][idx]
        ego_agent = torch.load(agent_path)
        env = init_env(layout=args.layout,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{meta_task}',
                       use_script_policy=True)
        save_path = f'../models/opponent/opponent_{args.layout}_{meta_task}.pth'
        model = BehaviorClone(state_dim=96, hidden_dim=128, action_dim=6).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        best_loss = 99999
        episodic_train_x = []
        episodic_train_y = []
        for i in range(data_collection_epochs):
            obs = env.reset()
            ego_obs, alt_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            eval = False
            if (i+1)%10 == 0:
                eval = True
            while not done:
                ai_act = evaluate(ego_agent, ego_obs)
                obs, sparse_reward, done, info = env.step((ai_act, 1))
                ego_obs, alt_obs = obs['both_agent_obs']
                alt_dire = info['joint_action'][1]
                alt_a = Action.INDEX_TO_ACTION.index(alt_dire)
                ep_reward += sparse_reward
                # env.render(interval=0.08)
                episodic_train_x.append(alt_obs)
                episodic_train_y.append(alt_a)
            # print(f'Ep {i+1}:', ep_reward)

        X = np.array(episodic_train_x) # (n_episode*episode_len, state_dim)# 将每一局的state合并
        y = np.array(episodic_train_y) # 将每一局的action合并
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                                torch.tensor(y_train, dtype=torch.int64).unsqueeze(dim=1).to(device)), shuffle=True, batch_size=128)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                              torch.tensor(y_val, dtype=torch.int64).unsqueeze(dim=1).to(device)), shuffle=True, batch_size=128)
        model = BehaviorClone(state_dim=96, hidden_dim=128, action_dim=6).to(device)
        train(args, train_loader, val_loader, model)
        torch.save(model.state_dict(), save_path) # 只保存模型的参数
