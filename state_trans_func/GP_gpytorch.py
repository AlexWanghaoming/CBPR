import os.path

import torch
import math
import gpytorch
import pickle
import numpy as np
# train_x = torch.linspace(0, 1, 100)
#
# train_y = torch.stack([
#     torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
#     torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
# ], -1)

# train_x = torch.rand(1000, 97)
# train_y = torch.randn(1000, 2)  # 1000 examples, 96 dimensions, 6 actions
device = 'cuda'

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, outdim):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=outdim
        ).to(device)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=outdim, rank=1
        ).to(device)

    def forward(self, x):
        x.to(device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    f_read = open('gp_trajs.pkl', 'rb')
    meta_task_trajs = pickle.load(f_read)
    f_read.close()

    epochs = 50
    action_dim = 6

    state_dim = 97
    layout = 'cramped_room'

    for key in meta_task_trajs:
        save_path = f'../models/gp/gp_{layout}_{key}_s_prime_r.pth'
        if os.path.exists(save_path):
            # print('sssssssssssssssssssssss')
            continue
        states_train = meta_task_trajs[key]['train_s'][:10000]
        actions_train = meta_task_trajs[key]['train_a'][:10000]
        rewards_train = np.reshape(meta_task_trajs[key]['train_r'], (-1, 1))[:10000]
        s_prime_train = meta_task_trajs[key]['train_s_'][:10000]

        actions_one_hot = np.eye(action_dim)[actions_train]
        # 拼接状态和动作
        s_a = np.hstack([states_train, actions_one_hot])
        s_a = torch.from_numpy(s_a).float().to(device)

        # rewards_train = torch.from_numpy(rewards_train).float().to(device)
        # s_prime_train = torch.from_numpy(s_prime_train).float().to(device)
        s_prime_r = np.hstack([s_prime_train, rewards_train])
        s_prime_r = torch.from_numpy(s_prime_r).float().to(device)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=state_dim).to(device)
        model = MultitaskGPModel(s_a, s_prime_r, likelihood, outdim=state_dim).to(device)

        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(epochs):
            optimizer.zero_grad()
            output = model(s_a)
            loss = -mll(output, s_prime_r)
            loss.backward()
            print(f'Key={key}, iter %d/%d - Loss: %.3f' % (i + 1, epochs, loss.item()))
            optimizer.step()

        torch.save(model.state_dict(), save_path)


