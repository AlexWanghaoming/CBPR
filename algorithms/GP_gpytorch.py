import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

# 1. 定义数据集
num_samples = 100
state_dim = 96
action_space = 6

# 随机生成状态和动作数据
states = torch.randn(num_samples, state_dim)
actions = torch.randint(0, action_space, (num_samples,))

# 将动作转换为独热编码
actions_onehot = torch.nn.functional.one_hot(actions, action_space).float()
inputs = torch.cat([states, actions_onehot], dim=-1)

# 这里只是模拟生成s'和r，你应该使用你的真实数据
next_states = states + 0.1 * torch.randn(num_samples, state_dim)
rewards = torch.randn(num_samples, 1)
outputs = torch.cat([next_states, rewards], dim=-1)

# 2. 定义高斯过程模型
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.size(-1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(inputs, outputs, likelihood)

# 3. 定义一个训练循环
def train(model, likelihood, inputs, outputs, epochs=100):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = -mll(output, outputs)
        loss.backward()
        optimizer.step()

# 4. 训练模型
train(model, likelihood, inputs, outputs)

# 5. 使用模型进行预测
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_states = torch.randn(2, state_dim)
    test_actions = torch.tensor([1, 4]) # 只是示例动作
    test_actions_onehot = torch.nn.functional.one_hot(test_actions, action_space).float()
    test_inputs = torch.cat([test_states, test_actions_onehot], dim=-1)
    observed_pred = likelihood(model(test_inputs))

print(observed_pred.mean)

