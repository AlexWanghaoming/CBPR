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
# from bc_hh_meta_task_key_state import train

import torch.optim as optim
device = torch.device("cpu")
from sklearn.cluster import KMeans

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 均值和方差
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # 假设输入数据在[0, 1]范围内
        )
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=2) #
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def save_encoder(self, path):
        torch.save(self.encoder, path)

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



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, default='marshmallow_experiment')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    opt = parser.parse_args()
    return opt


def slide_window(array:np.ndarray, window_size=20, step_size=5):
    overlap = window_size - step_size
    num_windows = (array.shape[0] - overlap) // step_size
    windows = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = array[start_idx:end_idx, :]
        windows.append(window)

    windows_array = np.array(windows)
    return windows_array

def _scale(data:np.ndarray):
    min_val = np.min(data, axis=0, keepdims=True)
    max_val = np.max(data, axis=0, keepdims=True)

    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    # scale
    data_normalized = (data - min_val) / range_val
    return data_normalized, min_val, max_val

def _norm(data:np.ndarray):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1
    # normalization
    data_normalized = (data - mean) / std
    return data_normalized, mean, std


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
    all_X = np.vstack(inputs) # (n_episode*episode_len, state_dim)
    all_y = np.vstack(targets)

    # all_X, mu, sigma = _norm(all_X)

    all_X, min_val, max_val = _scale(all_X)


    X = slide_window(all_X)
    y = slide_window(all_y)

    train_loader = DataLoader(torch.tensor(X).float(), shuffle=True, batch_size=64)

    # 超参数
    input_dim = X.shape[-1]  # 输入轨迹的维度
    hidden_dim = 256  # 隐藏层维度
    latent_dim = 64  # 潜在空间维度
    learning_rate = 1e-3
    epochs = 200

    # 初始化模型、优化器和损失函数
    vae_model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    # 训练VAE模型
    for epoch in range(epochs):
        for batch_data in train_loader:  # 假设data_loader是轨迹数据的迭代器
            optimizer.zero_grad()
            recon, mu, log_var = vae_model(batch_data)  # batch_data (B * SeqLen * feature_dim)
            MSE_loss = criterion(recon, batch_data)
            KLD_loss = 0.01 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = MSE_loss - KLD_loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    vae_model.save_encoder(path=f'../models/vae/{opt.layout}_vae_encoder.pth')


    # 提取轨迹的嵌入表示
    with torch.no_grad():
        no_overlap_trajs_data = slide_window(all_X, window_size=20, step_size=20)
        no_overlap_y = slide_window(all_y, window_size=20, step_size=20)

        y = torch.IntTensor(no_overlap_y)
        x = torch.FloatTensor(no_overlap_trajs_data)
        _, mu, _ = vae_model(x)  # 假设traj_data是需要提取嵌入表示的轨迹数据
        traj_embedding = mu.flatten(1).numpy()  # 转换为NumPy数组

    k = 4
    clusters = KMeans(n_clusters=k).fit(traj_embedding)
    for i in range(k):
        ## train BC model for each vae cluster
        dd = x[clusters.labels_ == i].view(-1, x.shape[-1]).detach().numpy()
        target = y[clusters.labels_ == i].view(-1, y.shape[-1]).detach().numpy()

        X_train, X_val, y_train, y_val = train_test_split(dd, target, test_size=0.15)

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

        save_path = f'../models/bc/BC_{opt.layout}_vae_cluster{i}.pth'
        torch.save(model, save_path)
