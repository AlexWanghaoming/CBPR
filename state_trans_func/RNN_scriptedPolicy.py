import torch
import pickle
import os, sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from My_utils import init_env
from models import MTP_MODELS, META_TASKS, SP_MODELS, BCP_MODELS
import random
import math
from src.overcooked_ai_py.mdp.actions import Action


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAYOUT_NAME = 'coordination_ring'
# LAYOUT_NAME = 'marshmallow_experiment'
INPUT_LENGTH = 30
PREDICT_LENGTH = 10
ACTION_DIM = 6
STATE_DIM = 96

class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=960):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.flatten_layer = nn.Flatten()
        # self.fc_1 = nn.Linear(INPUT_LENGTH * hidden_size, 1024)
        self.fc_1 = nn.Linear(hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc_2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x, _ = self.rnn(x) # (B, S, H)
        x = x[:, -1, :]  # 只使用最后一个时间步的隐藏状态
        # x = self.flatten_layer(x) # (B, S*H)
        x = F.relu(self.bn1(self.fc_1(x)))
        # x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(targets.size(0), -1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试函数
def val(model, test_loader, criterion):
    model.eval()
    var = torch.tensor([0.1])
    std_dev = torch.sqrt(var).to(device)
    total_loss = 0
    su_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            probs = (1 / (std_dev * (math.pi * 2) ** 0.5)) * torch.exp(
                -(targets.view(targets.size(0), -1) - outputs) ** 2 / (2 * std_dev ** 2))
            su = torch.sum(probs).item()
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            total_loss += loss.item()
            su_total += su
    print("mean_probs: ", su_total / len(test_loader))
    return total_loss / len(test_loader)


def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, initial_lr):
    """根据 epoch 调整学习率"""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    # print("lr:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def generate_dataset(N_samples, meta_task):
    num_episodes = 99999999
    input_seqs = []
    predict_seqs = []
    # mtp_model_path = MTP_MODELS[LAYOUT_NAME][idx]
    agent_path = BCP_MODELS[LAYOUT_NAME]   # bcp agent的能力最差， 和script policy agent 合作可以见到尽可能大的状态空间
    ego_agent = torch.load(agent_path)
    env = init_env(layout=LAYOUT_NAME,
                   agent0_policy_name='sp',
                   agent1_policy_name=f'script:{meta_task}',
                   use_script_policy=True)
    for _ in range(num_episodes):
        obs = env.reset()
        state_seq = []
        # action_seq = []
        ego_obs, alt_obs = obs['both_agent_obs']
        ep_reward = 0
        done = False
        while not done:
            ai_act = evaluate(ego_agent, ego_obs)
            obs, sparse_reward, done, info = env.step((ai_act, 1))
            ego_obs, alt_obs = obs['both_agent_obs']
            alt_dire = info['joint_action'][1]
            alt_a = Action.INDEX_TO_ACTION.index(alt_dire)
            alt_a_onehot = np.eye(ACTION_DIM)[alt_a]
            s_a = np.hstack([alt_obs, alt_a_onehot])
            state_seq.append(s_a)
            ep_reward += sparse_reward
            # env.render(interval=0.08)
        # print(f'Ep:', ep_reward)
        start_pos = random.sample(range(1, 600-INPUT_LENGTH-PREDICT_LENGTH), 100)
        for s in start_pos:
            input_seqs.append(state_seq[s:s+INPUT_LENGTH])
            predict_seqs.append(state_seq[s+INPUT_LENGTH: s+INPUT_LENGTH+PREDICT_LENGTH])
        if len(input_seqs) > N_samples:
            break
        else:
            # pass
            print(f'{len(input_seqs)} sequences collected')
    return np.array(input_seqs), np.array(predict_seqs)


def generate_data_or_load(N, meta_task, dataset_path):
    if not os.path.exists(dataset_path):
        # 生成训练和测试数据
        input_seqs, target_seqs = generate_dataset(N_samples=N, meta_task=meta_task)
        # 保存数据
        np.savez(dataset_path, input_seqs=input_seqs, predict_seqs=target_seqs)
    else:
        # 加载数据
        print('加载数据')
        dataset = np.load(dataset_path)
        input_seqs, target_seqs = dataset['input_seqs'], dataset['predict_seqs']
    return input_seqs, target_seqs
        
        
if __name__ == '__main__':
    # 生成测试数据
    test_input_seqs1, test_predict_seqs1 = generate_data_or_load(N=10000, meta_task=META_TASKS[LAYOUT_NAME][0],
                                                               dataset_path=f"{LAYOUT_NAME}_{META_TASKS[LAYOUT_NAME][0]}_test_rnn_dataset.npz")
    # test_input_seqs2, test_predict_seqs2 = generate_data_or_load(N=10000, meta_task=META_TASKS[LAYOUT_NAME][1],
    #                                                              dataset_path=f"{LAYOUT_NAME}_{META_TASKS[LAYOUT_NAME][1]}_test_rnn_dataset.npz")
    # test_input_seqs3, test_predict_seqs3 = generate_data_or_load(N=10000, meta_task=META_TASKS[LAYOUT_NAME][2],
    #                                                              dataset_path=f"{LAYOUT_NAME}_{META_TASKS[LAYOUT_NAME][2]}_test_rnn_dataset.npz")
    # test_input_seqs4, test_predict_seqs4 = generate_data_or_load(N=10000, meta_task=META_TASKS[LAYOUT_NAME][3],
    #                                                              dataset_path=f"{LAYOUT_NAME}_{META_TASKS[LAYOUT_NAME][3]}_test_rnn_dataset.npz")

    for idx, meta_task in enumerate(META_TASKS[LAYOUT_NAME]):
        metatask_idx_pool = [i for i in range(len(META_TASKS[LAYOUT_NAME]))]
        del metatask_idx_pool[idx]
        dataset_path = f"{LAYOUT_NAME}_{meta_task}_train_rnn_dataset.npz"
        train_input_seqs, train_predict_seqs = generate_data_or_load(N=50000, meta_task=meta_task, dataset_path=dataset_path)

        # numpy -> tensor
        train_sequences_tensor = torch.tensor(train_input_seqs, dtype=torch.float32).to(device)
        train_next_states_tensor = torch.tensor(train_predict_seqs[:, :, :96], dtype=torch.float32).to(device)
        test_sequences_tensor1 = torch.tensor(test_input_seqs1, dtype=torch.float32).to(device)
        test_next_states_tensor1 = torch.tensor(test_predict_seqs1[:, :, :96], dtype=torch.float32).to(device)

        # test_sequences_tensor2 = torch.tensor(test_input_seqs2, dtype=torch.float32).to(device)
        # test_next_states_tensor2 = torch.tensor(test_predict_seqs2[:, :, :96], dtype=torch.float32).to(device)
        # test_sequences_tensor3 = torch.tensor(test_input_seqs3, dtype=torch.float32).to(device)
        # test_next_states_tensor3 = torch.tensor(test_predict_seqs3[:, :, :96], dtype=torch.float32).to(device)
        # test_sequences_tensor4 = torch.tensor(test_input_seqs4, dtype=torch.float32).to(device)
        # test_next_states_tensor4 = torch.tensor(test_predict_seqs4[:, :, :96], dtype=torch.float32).to(device)

        # 创建 DataLoader
        train_dataset = TensorDataset(train_sequences_tensor, train_next_states_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_dataset1 = TensorDataset(test_sequences_tensor1, test_next_states_tensor1)
        test_loader1 = DataLoader(test_dataset1, batch_size=256)
        # test_dataset2 = TensorDataset(test_sequences_tensor2, test_next_states_tensor2)
        # test_loader2 = DataLoader(test_dataset2, batch_size=256)
        # test_dataset3 = TensorDataset(test_sequences_tensor3, test_next_states_tensor3)
        # test_loader3 = DataLoader(test_dataset3, batch_size=256)
        # test_dataset4 = TensorDataset(test_sequences_tensor4, test_next_states_tensor4)
        # test_loader4 = DataLoader(test_dataset4, batch_size=256)

        # 初始化 RNN 模型
        state_action_dim = STATE_DIM + ACTION_DIM  # state_dim
        hidden_size = 128
        num_layers = 1
        output_size = STATE_DIM * PREDICT_LENGTH  # 预测 10 个状态，每个状态 96 个特征
        model = RNNPredictor(input_size=state_action_dim, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print('Start traning RNN model')
        num_epochs = 50
        warmup_epochs = 3
        initial_lr = 0.001
        save_path = f'../models/RNN/rnn_{LAYOUT_NAME}_{meta_task}_s_prime_r.pth'
        for epoch in range(num_epochs):
            print('--------------------------------------------------------------------------')
            adjust_learning_rate(optimizer, epoch, warmup_epochs, num_epochs, initial_lr)
            train_loss = train(model, train_loader, optimizer, criterion)

            # 在4个metatask生成的测试集合上分别测试
            # for mts, test_loader in enumerate([test_loader1, test_loader2, test_loader3, test_loader4]):
            for mts, test_loader in enumerate([test_loader1]):
                test_loss = val(model, test_loader, criterion)
                if mts == idx:
                    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
                else:
                    print(f'mt{mts+1} test loss: {test_loss:.4f}')
        torch.save(model.state_dict(), save_path)