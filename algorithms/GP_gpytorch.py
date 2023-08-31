sssss
import torch
import gpytorch

class GPStateTransitionModel:
    class CustomGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __init__(self, action_dim=6, state_dim=96):
        self.model_s_prime = None
        self.model_r = None
        self.likelihood_s_prime = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood_r = gpytorch.likelihoods.GaussianLikelihood()
        self.action_dim = action_dim
        self.state_dim = state_dim

    def train(self, states, actions, s_prime, rewards, epochs=100, learning_rate=0.1):
        actions_one_hot = torch.eye(self.action_dim)[actions]
        s_a = torch.cat([states, actions_one_hot], dim=1)

        self.model_s_prime = self.CustomGP(s_a, s_prime, self.likelihood_s_prime)
        self.model_r = self.CustomGP(s_a, rewards, self.likelihood_r)

        optimizer = torch.optim.Adam([
            {'params': self.model_s_prime.parameters()},
            {'params': self.model_r.parameters()}
        ], lr=learning_rate)

        mll_s_prime = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_s_prime, self.model_s_prime)
        mll_r = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_r, self.model_r)

        self.model_s_prime.train()
        self.model_r.train()
        self.likelihood_s_prime.train()
        self.likelihood_r.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output_s_prime = self.model_s_prime(s_a)
            output_r = self.model_r(s_a)
            pred_s_prime = self.likelihood_s_prime(output_s_prime)
            pred_r = self.likelihood_r(output_r)
            loss = -mll_s_prime(pred_s_prime, s_prime) - mll_r(pred_r, rewards)
            loss.backward()
            optimizer.step()

    def predict(self, states, actions):
        actions_one_hot = torch.eye(self.action_dim)[actions]
        s_a = torch.cat([states, actions_one_hot], dim=1)

        self.model_s_prime.eval()
        self.model_r.eval()
        self.likelihood_s_prime.eval()
        self.likelihood_r.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            s_prime_pred = self.likelihood_s_prime(self.model_s_prime(s_a))
            r_pred = self.likelihood_r(self.model_r(s_a))

        return s_prime_pred.mean, r_pred.mean

# 示例
if __name__ == "__main__":
    states_train = torch.rand(100, 96)
    actions_train = torch.randint(0, 6, (100,))
    s_prime_train = torch.rand(100, 96)
    rewards_train = torch.rand(100, 1)

    model = GPStateTransitionModel()
    model.train(states_train, actions_train, s_prime_train, rewards_train)

    states_test = torch.rand(5, 96)
    actions_test = torch.randint(0, 6, (5,))
    s_prime_pred, r_pred = model.predict(states_test, actions_test)
    print("Predicted s':", s_prime_pred)
    print("Predicted r:", r_pred)

