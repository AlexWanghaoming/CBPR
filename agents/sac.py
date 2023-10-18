"""
REF: https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC14%E7%AB%A0-SAC%E7%AE%97%E6%B3%95.ipynb
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from utils import get_device

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/stable-baselines/')
from typing import Union


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC:
    ''' 离散SAC '''
    def __init__(self,
                 env,
                 hidden_dim,
                 lr,
                 tau: float = 0.005,
                 adaptive_alpha:bool = False,
                 clip_grad_norm:float = 0.5,
                 use_lr_decay:bool = False,
                 device: Union[torch.device, str] = 'auto'):

        self.device = get_device(device)
        self.use_lr_decay = use_lr_decay
        # 策略网络
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(self.device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(self.device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.adaptive_alpha = adaptive_alpha  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
            ## elegantRL
            self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
            self.alpha_optimizer = torch.optim.AdamW((self.alpha_log,), lr=lr)
        else:
            self.alpha = 0.1  # 0.005-0.5
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.lr = lr
        self.gamma = 0.99
        self.tau = tau
        self.clip_grad_norm = clip_grad_norm

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # print(action)
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def _calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)

        alpha = self.alpha_log.exp().detach()
        next_qvalue = min_qvalue + alpha * entropy

        td_target = rewards + self.gamma * next_qvalue * (1 - dones)
        return td_target

    def update(self, transition_dict, t_env):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self._calc_target(rewards, next_states, dones)

        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_norm)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_norm)
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        # print("entropy:", entropy)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 根据概率计算期望
        ## ElegantRL
        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

        actor_loss = torch.mean(-alpha * entropy - min_qvalue)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()

        # 更新alpha值
        if self.adaptive_alpha:
            alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.alpha_log)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        ## 打印网络的梯度
        # print_grads(self.actor)
        # print_grads(self.critic_1)
        # print_grads(self.critic_2)
        ## 打印loss
        # print('actor_loss:',actor_loss.item())
        # print('critic1_loss:',critic_1_loss.item())
        # print('critic2_loss:',critic_2_loss.item())
        # print('alpha_loss:',alpha_loss.item())

        # 软更新target critic
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(t_env)

    def _soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def lr_decay(self, cur_steps):
        factor =  max(1 - cur_steps/900000, 0.33333)
        lr_a_now = self.lr * factor
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_1_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_2_optimizer.param_groups:
            p['lr'] = lr_a_now



import collections
import random

class OffpolicyReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if self.size() < batch_size:
            transitions = random.sample(self.buffer, self.size())
        else:
            transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
