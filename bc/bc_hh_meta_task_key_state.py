import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src/')
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2020_HUMAN_DATA_TRAIN
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import argparse
from typing import Dict, Tuple, List
from bc_hh import BehaviorClone

device = torch.device("cpu")

def train(args, train_loader, val_loader, model):
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
        
        
@torch.no_grad()
def _val_in_one_epoch(val_loader, model, stochastic=True):
    model.eval()
    losses = []
    pred = []
    y_target = []
    for data in val_loader:
        x, labels = data
        logits, loss = model(x, labels)
        probs = F.softmax(logits)
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


def group_by_columns_with_indices(matrix:np.ndarray, columns:slice) -> Tuple[Dict[Tuple, np.ndarray], Dict[Tuple, List]]:
    groups = {}
    indices = {}
    for i, row in enumerate(matrix):
        # 将指定列的值转换为元组作为键
        key = tuple(row[columns])
        if key not in groups:
            groups[key] = []
            indices[key] = []
        groups[key].append(row)
        indices[key].append(i)
    # 将每个组的列表转换为numpy数组
    for key in groups:
        groups[key] = np.array(groups[key])
    return groups, indices


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, default='marshmallow_experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    DEFAULT_DATA_PARAMS = {
        "layouts": [opt.layout],

        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2020_HUMAN_DATA_TRAIN,
    }
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    inputs, targets = (processed_trajs["ep_states"], processed_trajs["ep_actions"])
    # 将每一局的轨迹合并
    all_X = np.vstack(inputs) # (n_episode*episode_len, state_dim)
    all_y = np.vstack(targets)
    
    ## wanghm 根据key state对智能体的轨迹进行划分
    columns_to_group_by = slice(4, 8)  # 状态空间的5-8维通过one-hot编码智能体手持物体
    X_groups, group_indices = group_by_columns_with_indices(all_X, columns_to_group_by)
    # 使用分组索引对第二个矩阵进行分组
    y_groups = {key: all_y[indices] for key, indices in group_indices.items()}
    
    ## wanghm 根据不同的meta-task 训练不同的 bc models
    for key in X_groups:
        print(f"Group {key} has {X_groups[key].shape[0]} samples in all_X and {y_groups[key].shape[0]} samples in all_y.")
        X = X_groups[key]
        y = y_groups[key]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(),
                                                torch.tensor(y_train, dtype=torch.int64)),
                                  shuffle=True,
                                  batch_size=64)

        val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(),
                                              torch.tensor(y_val, dtype=torch.int64)),
                                shuffle=True,
                                batch_size=64)

        model = BehaviorClone(state_dim=96, hidden_dim=32, action_dim=6)
        model.to(device)
        train(opt, train_loader, val_loader, model)

        save_path = f'../models/bc/BC_{opt.layout}_{key}.pth'
        torch.save(model, save_path)
