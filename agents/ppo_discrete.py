import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)


class MlpActor(nn.Module):
    def __init__(self, args):
        super(MlpActor, self).__init__()
        self.net_arch = args.net_arch

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = nn.ReLU()
        layer_init_with_orthogonal(self.fc1)
        layer_init_with_orthogonal(self.fc2)
        layer_init_with_orthogonal(self.fc3)

        self.state_avg = nn.Parameter(torch.zeros((args.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((args.state_dim,)), requires_grad=False)

    def forward(self, s):
        s = self.state_norm(s)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)

        return a_prob

    def state_norm(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_avg) / self.state_std

class MlpCritic(nn.Module):
    def __init__(self, args):
        super(MlpCritic, self).__init__()
        self.net_arch = args.net_arch

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        self.activate_func = nn.ReLU() #
        layer_init_with_orthogonal(self.fc1)
        layer_init_with_orthogonal(self.fc2)
        layer_init_with_orthogonal(self.fc3)

        self.state_avg = nn.Parameter(torch.zeros((args.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((args.state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)

        values = self.value_re_norm(v_s)
        return values


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
        self.use_lr_decay = args.use_lr_decay
        self.vf_coef = args.vf_coef

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

            reward_sums = adv + vs

            adv = ((adv - adv.mean()) / (adv.std() + 1e-8))   # Trick 1:advantage normalization

            self.update_avg_std_for_normalization(
                states=s.reshape((-1, self.args.state_dim)),
                returns=reward_sums.reshape((-1,))
            )
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
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
                # self.optimizer_actor.step()
                # Update critic
                # critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm)
                # self.optimizer_critic.step()

                self.optimizer.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(cur_steps)

    def update_avg_std_for_normalization(self, states: torch.Tensor, returns: torch.Tensor):
        # tau = self.state_value_tau
        tau = self.args.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.actor.state_avg[:] = self.actor.state_avg * (1 - tau) + state_avg * tau
        self.actor.state_std[:] = self.critic.state_std * (1 - tau) + state_std * tau + 1e-4
        self.critic.state_avg[:] = self.actor.state_avg
        self.critic.state_std[:] = self.actor.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.critic.value_avg[:] = self.critic.value_avg * (1 - tau) + returns_avg * tau
        self.critic.value_std[:] = self.critic.value_std * (1 - tau) + returns_std * tau + 1e-4

    def lr_decay(self, cur_steps):
        factor =  max(1 - cur_steps/self.args.t_max, 0.33333)
        lr_a_now = self.lr * factor
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