"""
对手建模：Accurate policy detection and efficient knowledge reuse against muti-strategic opponents (2022, KBS )
"""
import torch
import torch.nn as nn
import math
import os, sys
import numpy as np
import torch.nn.functional as F
from My_utils import init_env
from models import MTP_MODELS, META_TASKS, BCP_MODELS, SP_MODELS
import random
from src.overcooked_ai_py.mdp.actions import Action


device = 'cuda'
LAYOUT_NAME = 'soup_coordination'
# LAYOUT_NAME = 'asymmetric_advantages'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, initial_lr):
    """根据 epoch 调整学习率"""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    # print("lr:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Opponent(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal=True):
        super(Opponent, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(init_(nn.Linear(state_dim, hidden_dim)),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(init_(nn.Linear(hidden_dim, hidden_dim)),
                                 nn.ReLU())
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, target):  # obs: (bs, state_dim), target: (bs, 1)
        logits = self.fc3(self.fc2(self.fc1(obs)))
        probs = F.softmax(logits)
        probs = probs.gather(1, target)
        return probs


def evaluate(actor, s):
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    a_prob = actor(s).detach().cpu().numpy().flatten()
    a = np.argmax(a_prob)
    return a


def log_prob_loss(model, input, target):
    # eta = 0.001
    eta = 0.001
    probs = model(input, target.unsqueeze(dim=1))
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs)
    loss = -torch.mean(log_probs + eta*entropy)
    return loss


if __name__ == '__main__':
    num_epochs = 100
    warmup_epochs = 2
    initial_lr = 0.001
    for idx, meta_task in enumerate(META_TASKS[LAYOUT_NAME]):
        # agent_path = BCP_MODELS[LAYOUT_NAME]  # bcp agent的能力最差， 和script policy agent 合作可以见到尽可能大的状态空间
        agent_path = MTP_MODELS[LAYOUT_NAME][idx]

        ego_agent = torch.load(agent_path)
        env = init_env(layout=LAYOUT_NAME,
                       agent0_policy_name='mtp',
                       agent1_policy_name=f'script:{meta_task}',
                       use_script_policy=True)
        save_path = f'../models/opponent/opponent_{LAYOUT_NAME}_{meta_task}.pth'
        if os.path.exists(save_path):
            continue
        model = Opponent(state_dim=96, hidden_dim=128, action_dim=6).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        best_loss = 99999
        for i in range(num_epochs):
            adjust_learning_rate(optimizer, i, warmup_epochs, num_epochs, initial_lr)
            obs = env.reset()
            ego_obs, alt_obs = obs['both_agent_obs']
            ep_reward = 0
            done = False
            episodic_train_x = []
            episodic_train_y = []
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
            print(f'Ep {i+1}:', ep_reward)
            input_tensor = torch.tensor(episodic_train_x, dtype=torch.float32).to(device)
            out_tensor = torch.tensor(episodic_train_y, dtype=torch.int64).to(device)
            if eval:
                model.eval()
                val_loss = log_prob_loss(model, input_tensor, out_tensor).item()
                print(f'Meta-task: {meta_task}, iter %d/%d - Test loss: %.3f' % (i + 1, num_epochs, val_loss))
                if val_loss < best_loss:
                    print(f'saving opponent model {save_path}')
                    torch.save(model.state_dict(), save_path)
                    best_loss = val_loss
            else:
                train_loss = log_prob_loss(model, input_tensor, out_tensor)
                train_loss.backward()
                optimizer.step()
                print(f'Meta-task: {meta_task}, iter %d/%d - Train loss: %.3f' % (i + 1, num_epochs, train_loss.item()))

