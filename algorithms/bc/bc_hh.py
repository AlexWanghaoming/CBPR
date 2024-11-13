import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../src/')
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2020_HUMAN_DATA_TRAIN
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
from typing import *
import argparse
import math

device = torch.device("cpu")


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class BehaviorClone(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal=True):
        super(BehaviorClone, self).__init__()
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.fc1 = nn.Sequential(init_(nn.Linear(state_dim, hidden_dim)),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(init_(nn.Linear(hidden_dim, hidden_dim)),
                                 nn.ReLU())
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, obs, target): # obs: (bs, state_dim), target: (bs, 1)
        logits = self.fc3(self.fc2(self.fc1(obs)))
        loss = self.loss_fn(logits, target.squeeze(axis=1))
        
        ## wanghm: 下面的loss计算方法和nn.CrossEntropyLoss()等效
        # probs = F.softmax(logits)
        # log_probs = torch.log(probs.gather(1, target))
        # loss = torch.mean(-log_probs)
        
        return logits, loss

    def choose_action(self, obs, deterministic=True): # obs: (state_dim)
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        # obs = obs.to(self.device)
        with torch.no_grad():
            logits = self.fc3(self.fc2(self.fc1(obs)))
            a_prob = F.softmax(logits, dim=1)
            if deterministic:
                a = np.argmax(a_prob.detach().cpu().numpy().flatten())
            else:
                dist = Categorical(probs=a_prob)
                a = dist.sample().cpu().numpy()[0]
            return a

    def action_probability(self, obs):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        # obs = obs.to(self.device)
        with torch.no_grad():
            logits = self.fc3(self.fc2(self.fc1(obs)))
            a_prob = F.softmax(logits, dim=1)
            return a_prob.detach().cpu().numpy().flatten()

        
def train(args, train_loader, val_loader, model, group_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    for k in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for x, label in train_loader:
            optimizer.zero_grad()
            x.to(device)
            label.to(device)
            # _, loss = model(x, label)
            _, loss = model(x, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # lr_scheduler.step()
        train_loss = sum(train_loss)/len(train_loss)
        
        val_loss, val_accuracy = _val_in_one_epoch(val_loader, model, stochastic=False)
        print({'epoch':k, 'training_loss':train_loss,'val_loss':val_loss, 'val_accuracy':val_accuracy})
        if group_name == 'HP':
            if k%10 == 0:
                save_path = f'../models/bc/{group_name}_{opt.layout}_{k}_epoch.pth'
                torch.save(model, save_path) # 保存整个模型
        else:
            if k == args.epochs:
                save_path = f'../models/bc/{group_name}_{opt.layout}.pth'
                torch.save(model, save_path)  # 保存整个模型
@torch.no_grad()
def _val_in_one_epoch(val_loader, model, stochastic=True):
    model.eval()
    losses = []
    pred = []
    y_target = []
    for data in val_loader:
        x, labels = data
        logits, loss = model(x, labels)
        probs = F.softmax(logits, dim=1)
        if stochastic:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        else:
            action = torch.argmax(probs, dim=1)
        pred.extend(action.tolist())
        y_target.extend(labels.squeeze(axis=1).tolist())
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    accucary = sum(np.array(pred) == np.array(y_target)) / len(pred)
    return val_loss, accucary


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--layout', type=str, default='random3')
    parser.add_argument('--layout', type=str, default='soup_coordination')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-4)
    opt = parser.parse_args()
    return opt


def shuff_and_split(list1, list2) -> Tuple[zip, zip]:
    # 将两个列表zip在一起，形成元素对
    zipped_list = list(zip(list1, list2))
    # 打乱这个元素对列表
    random.shuffle(zipped_list)
    # 将打乱后的元素对列表分成两半
    midpoint = len(zipped_list) // 2
    first_half = zipped_list[:midpoint]
    second_half = zipped_list[midpoint:]
    return zip(*first_half), zip(*second_half)


if __name__ == '__main__':
    opt = parse_opt()
    DEFAULT_DATA_PARAMS = {
        # "layouts": ['cramped_room'],
        # "layouts": ['asymmetric_advantages'],
        # "layouts": ['coordination_ring'],
        # "layouts": ['random0'],
        "layouts": [opt.layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2019_HUMAN_DATA_TRAIN if opt.layout in ['cramped_room',
                                                                   'asymmetric_advantages',
                                                                   "coordination_ring",
                                                                   'random0',
                                                                   'random3'] else CLEAN_2020_HUMAN_DATA_TRAIN,
    }
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    inputs, targets = (processed_trajs["ep_states"], processed_trajs["ep_actions"])
    id = 0
    group = ['BC', 'HP']
    for states, actions in shuff_and_split(inputs, targets):
        group_name = group[id]
        X = np.vstack(states) # (n_episode*episode_len, state_dim)# 将每一局的state合并
        y = np.vstack(actions) # 将每一局的action合并
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(),
                                                torch.tensor(y_train, dtype=torch.int64)), shuffle=True, batch_size=64)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(),
                                              torch.tensor(y_val, dtype=torch.int64)), shuffle=True, batch_size=64)
        model = BehaviorClone(state_dim=96, hidden_dim=64, action_dim=6).to(device)
        train(opt, train_loader, val_loader, model, group_name)
        id = id + 1
