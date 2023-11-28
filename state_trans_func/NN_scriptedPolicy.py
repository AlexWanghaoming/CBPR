import torch
import torch.nn as nn
import pickle
import os, sys
import numpy as np
import torch.nn.functional as F
from My_utils import init_env
from models import MTP_MODELS, META_TASKS
from tqdm import tqdm
import random
from src.overcooked_ai_py.mdp.actions import Action


device = 'cuda'
LAYOUT_NAME = 'cramped_room'
# LAYOUT_NAME = 'marshmallow_experiment'


class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activate_func = nn.ReLU()

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        output = self.fc3(s)
        return output

    def action_probability(self, obs):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        # obs = obs.to(self.device)
        with torch.no_grad():
            logits = self.fc3(self.fc2(self.fc1(obs)))
            a_prob = F.softmax(logits, dim=1)
            return a_prob.detach().cpu().numpy().flatten()


@torch.no_grad()
def val(model, x_val,  y_val):
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    val_loss = loss_fn(model(x_val), y_val)
    return val_loss


def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


def shuffle_dict(dictOfList):
    # 获取列表长度
    list_length = len(dictOfList['train_s'])  # 假设所有列表长度相同
    # 创建一个索引列表
    index_list = list(range(list_length))
    # 打乱索引列表
    random.shuffle(index_list)
    # 使用打乱的索引列表来重新排列字典中的每个列表
    for key in dictOfList.keys():
        dictOfList[key] = [dictOfList[key][i] for i in index_list]


if __name__ == '__main__':
    N = 100000
    num_episodes = N//600 + 1
    for idx, meta_task in enumerate(META_TASKS[LAYOUT_NAME]):
        # meta_task_trajs = {'train_s': [], 'train_a': [], 'train_r': [], 'train_s_': []}
        meta_task_trajs = {'train_s': [], 'train_a': [], 'train_s_': []}
        mtp_model_path = MTP_MODELS[LAYOUT_NAME][idx]
        print('mtp model path:',mtp_model_path )
        mtp_agent = torch.load(mtp_model_path)
        env = init_env(layout=LAYOUT_NAME,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{meta_task}',
                       use_script_policy=True)
        print(f'Collecting training data ... for {meta_task}')
        for _ in tqdm(range(num_episodes)):
            obs = env.reset()
            ego_obs, alt_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            while not done:
                ai_act = evaluate(mtp_agent, ego_obs)
                obs_, sparse_reward, done, info = env.step((ai_act, 1))
                ego_obs_, alt_obs_ = obs_['both_agent_obs']
                alt_dire = info['joint_action'][1]
                alt_a = Action.INDEX_TO_ACTION.index(alt_dire)
                ep_reward += sparse_reward
                if len(meta_task_trajs['train_s']) <= N:
                    meta_task_trajs['train_s'].append(alt_obs)
                    meta_task_trajs['train_a'].append(alt_a)
                    # meta_task_trajs['train_r'].append(sparse_reward)
                    meta_task_trajs['train_s_'].append(alt_obs_)
                else:
                    break
                ego_obs, alt_obs = ego_obs_, alt_obs_
                # env.render(interval=0.08)
            print(f'Ep:', ep_reward)

        shuffle_dict(meta_task_trajs) # 将样本打乱，方便神经网络学习

        for i in meta_task_trajs:
            meta_task_trajs[i] = np.array(meta_task_trajs[i])

        print('Start traning NN model')
        epochs = 500
        action_dim = 6
        loss_fn = torch.nn.MSELoss(reduction='mean')
        save_path = f'../models/NN/NN_{LAYOUT_NAME}_{meta_task}_s_prime_r.pth'
        states_train = meta_task_trajs['train_s']
        actions_train = meta_task_trajs['train_a']
        # rewards_train = np.reshape(meta_task_trajs['train_r'], (-1, 1))
        s_prime_train = meta_task_trajs['train_s_']
        actions_one_hot = np.eye(action_dim)[actions_train]
        # 拼接状态和动作
        s_a = np.hstack([states_train, actions_one_hot])
        s_a = torch.from_numpy(s_a).float().to(device)

        # rewards_train = torch.from_numpy(rewards_train).float().to(device)
        s_prime_train = torch.from_numpy(s_prime_train).float().to(device)
        # s_prime_r = torch.hstack([s_prime_train, rewards_train])


        model = NN(input_dim=s_a.shape[1], output_dim=s_prime_train.shape[1]).to(device)
        model.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(s_a)
            train_loss = loss_fn(output, s_prime_train)
            train_loss.backward()
            optimizer.step()
            # val_loss = val(model, x_val=s_a_test, y_val=s_prime_r_test)
            # print(f'Key={key}, iter %d/%d - Train loss: %.3f - Val loss: %.3f' % (i + 1, epochs, train_loss.item(), val_loss.item()))
            print(f'Meta-task: {meta_task}, iter %d/%d - Train loss: %.3f' % (i + 1, epochs, train_loss.item()))

        torch.save(model.state_dict(), save_path)

