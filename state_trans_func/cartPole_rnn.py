import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

env = gym.make('CartPole-v1')
input_sequence_length = 30
predict_sequence_length = 10

def generate_dataset(num_sequences):
    sequences = []
    next_states = []

    for _ in range(num_sequences):
        env.reset()
        seq = []
        for _ in range(input_sequence_length + predict_sequence_length):
            state, _, done, _ = env.step(env.action_space.sample())
            seq.append(state)
            if done:
                break
        if len(seq) == input_sequence_length + predict_sequence_length:
            sequences.append(seq[:input_sequence_length])
            next_states.append(seq[input_sequence_length:])

    return np.array(sequences), np.array(next_states)

# 生成数据集
train_sequences, train_next_states = generate_dataset(1000)
test_sequences, test_next_states = generate_dataset(200)

# 转换为 PyTorch 张量
train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.float32)
train_next_states_tensor = torch.tensor(train_next_states, dtype=torch.float32)
test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32)
test_next_states_tensor = torch.tensor(test_next_states, dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(train_sequences_tensor, train_next_states_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_sequences_tensor, test_next_states_tensor)
test_loader = DataLoader(test_dataset, batch_size=32)

class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # 只使用最后一个时间步的隐藏状态
        return x

# 初始化 RNN 模型
input_size = 4  # CartPole 状态维度
hidden_size = 64
num_layers = 1
output_size = 4 * predict_sequence_length  # 预测 10 个状态，每个状态 4 个特征
model = RNNPredictor(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
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
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 训练和测试循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = val(model, test_loader, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
