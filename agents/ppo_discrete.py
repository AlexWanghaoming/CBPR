import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)



class MlpActor(nn.Module):
    def __init__(self, args):
        super(MlpActor, self).__init__()
        self.net_arch = args.net_arch

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)

        return a_prob


class MlpCritic(nn.Module):
    def __init__(self, args):
        super(MlpCritic, self).__init__()
        self.net_arch = args.net_arch

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class SharedFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(SharedFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(args.state_dim, out_channels=25, kernel_size=5, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(25, out_channels=25, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(25, out_channels=25, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(500, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
        )

    def forward(self, s):  # input state (b, 5,4,20)
        s = s.permute(0, 3, 1, 2)  # -> (b, 20, 5, 4)
        s = self.conv_layers(s)
        return s


class CnnMlpActor(nn.Module):
    def __init__(self, args, feature_extractor):
        super(CnnMlpActor, self).__init__()
        self.net_arch = args.net_arch
        # self.feature_extractor = feature_extractor
        self.feature_extractor = SharedFeatureExtractor(args)
        # self.fc1 = nn.Linear(500, args.hidden_width)
        # self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            # orthogonal_init(self.fc1)
            # orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.feature_extractor(s)
        # s = self.activate_func(self.fc1(s))
        # s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)

        return a_prob


class CnnMlpCritic(nn.Module):
    def __init__(self, args, feature_extractor):
        super(CnnMlpCritic, self).__init__()
        self.net_arch = args.net_arch
        # self.feature_extractor = feature_extractor
        self.feature_extractor = SharedFeatureExtractor(args)
        # self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        # self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            # orthogonal_init(self.fc1)
            # orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.feature_extractor(s)
        # s = self.activate_func(self.fc1(s))
        # s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        # v_s = self.fc1(s)
        return v_s


class PPO_discrete:
    def __init__(self, args):
        self.args = args
        self.net_arch = args.net_arch
        self.device = args.device
        self.batch_size = args.batch_size
        # self.mini_batch_size = args.mini_batch_size
        self.mini_batch_size = args.mini_batch_size if args.use_minibatch else args.batch_size
        self.lr = args.lr  # Learning rate of actor
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.vf_coef = args.vf_coef

        if self.net_arch == "conv":
            self.feature_extractor = SharedFeatureExtractor(self.args)
            self.actor = CnnMlpActor(args, self.feature_extractor)
            self.critic = CnnMlpCritic(args, self.feature_extractor)
        else:
            self.actor = MlpActor(args)
            self.critic = MlpCritic(args)

        self.actor.to(self.device)
        self.critic.to(self.device)

        all_parameters = list(set(list(self.actor.parameters()) + list(self.critic.parameters())))
        self.optimizer = torch.optim.Adam(all_parameters, lr=self.lr, eps=1e-8)

        # PBT use
        self.total_steps = 0

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s = s.to(self.device)
        a_prob = self.actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s = s.to(self.device)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def update(self, replay_buffer, cur_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        s = s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        dw = dw.to(self.device)
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)  # 对应原文公式11
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                # self.optimizer_actor.zero_grad()
                # self.optimizer_critic.zero_grad()
                self.optimizer.zero_grad()

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]  # clip
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                #
                # print("actor loss:", torch.mean(actor_loss))
                # print("critic loss:", torch.mean(critic_loss))

                loss = actor_loss.mean() + self.vf_coef * critic_loss
                loss.backward()
                # Update actor
                # actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                # self.optimizer_actor.step()
                # Update critic
                # critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
                # self.optimizer_critic.step()

                self.optimizer.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(cur_steps)

    def lr_decay(self, cur_steps):
        factor =  max(1 - cur_steps/self.args.t_max, 0.33333)
    
        lr_a_now = self.lr * factor
        lr_c_now = self.lr * factor
    
        # for p in self.optimizer_actor.param_groups:
        #     p['lr'] = lr_a_now
        # for p in self.optimizer_critic.param_groups:
        #     p['lr'] = lr_c_now
        for p in self.optimizer.param_groups:
            p['lr'] = lr_a_now

    def save_actor(self, path='ppo_actor.pth'):
        torch.save(self.actor, path)

    def save_critic(self, path='ppo_critic.pth'):
        torch.save(self.critic, path)

    def load_actor(self, model_path):
        self.actor = torch.load(model_path, map_location=self.device)

    def load_critic(self, model_path):
        self.critic = torch.load(model_path, map_location=self.device)