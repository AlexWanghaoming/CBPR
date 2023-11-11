import torch
import torch.nn as nn
import pickle
import numpy as np
device = 'cuda'


class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.activate_func = nn.ReLU()

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        output = self.fc3(s)
        return output

@torch.no_grad()
def val(model, x_val,  y_val):
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    val_loss = loss_fn(model(x_val), y_val)
    return val_loss


if __name__ == '__main__':
    # LAYOUT_NAME = 'marshmallow_experiment'
    LAYOUT_NAME = 'cramped_room'
    f_read = open(f'/alpha/overcooked_rl/state_trans_func/gp_trajs_{LAYOUT_NAME}.pkl', 'rb')
    meta_task_trajs = pickle.load(f_read)
    f_read.close()
    epochs = 500
    action_dim = 6
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for key in meta_task_trajs:
        save_path = f'../models/NN/NN_{LAYOUT_NAME}_{key}_s_prime_r.pth'
        states_train = meta_task_trajs[key]['train_s'][:50000]
        actions_train = meta_task_trajs[key]['train_a'][:50000]
        rewards_train = np.reshape(meta_task_trajs[key]['train_r'], (-1, 1))[:50000]
        s_prime_train = meta_task_trajs[key]['train_s_'][:50000]
        actions_one_hot = np.eye(action_dim)[actions_train]
        # 拼接状态和动作
        s_a = np.hstack([states_train, actions_one_hot])
        s_a = torch.from_numpy(s_a).float().to(device)

        rewards_train = torch.from_numpy(rewards_train).float().to(device)
        s_prime_train = torch.from_numpy(s_prime_train).float().to(device)
        s_prime_r = torch.hstack([s_prime_train, rewards_train])

        # s_prime_r = torch.from_numpy(s_prime_train).float().to(device)  # s,a -> s_prime
        # s_a_test = s_a[:10000]
        # s_prime_r_test = s_prime_r[:10000]
        # s_a_train = s_a[10000:50000]
        # s_prime_r_train = s_prime_r[10000:50000]

        model = NN(input_dim=s_a.shape[1], output_dim=s_prime_r.shape[1]).to(device)
        model.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(s_a)
            train_loss = loss_fn(output, s_prime_r)
            train_loss.backward()
            optimizer.step()
            # val_loss = val(model, x_val=s_a_test, y_val=s_prime_r_test)
            # print(f'Key={key}, iter %d/%d - Train loss: %.3f - Val loss: %.3f' % (i + 1, epochs, train_loss.item(), val_loss.item()))
            print(f'Key={key}, iter %d/%d - Train loss: %.3f' % (i + 1, epochs, train_loss.item()))

        torch.save(model.state_dict(), save_path)

