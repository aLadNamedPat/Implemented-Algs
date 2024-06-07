import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Q_Net(nn.Module):    
    def __init__(self, obs_dims, num_actions, hidden_dim1 = 64, hidden_dim2 = 64, lr = 0.001):
        super(Q_Net, self).__init__()
        #Only the state is input because the action will be accounted for by the learning process of the agent
        self.Q_Net = nn.Sequential(
            nn.Linear(obs_dims, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, num_actions)
        )

        #Output is the number of possible actions that both could take
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, obs, mask = None):
        action = self.Q_Net(obs)
        if mask is not None:
            for i in range(len(mask)):
                if mask[i] == 0:
                    action[i] = float("-inf")
        return action
    
    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.optimizer.step()

class Q_net_LSTM(nn.Module):
    def __init__(self, input_size, num_actions, hidden_layer1 = 64, hidden_layer2 = 64, hidden_size = 16, num_layers = 1, lr = 0.001):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.lstm = nn.LSTM(hidden_layer2, num_actions, num_layers = num_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, obs, mask = None, hidden = None):
        if hidden is None:
            hidden = np.zeros(self.input_size)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x, h = self.lstm(x, hidden)
        if mask is not None:
            for i in range(len(mask)):
                if mask[i] == 0:
                    x[i] = float("-inf")
        return x, h

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Q_Net_Conv(nn.Module):
    def __init__(self, grid_length, kernel_size, hidden_size = 16, num_layers = 1, lr = 0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size)         #Size of the input becomes grid_length - kernel_size
        self.pool = nn.MaxPool2d(2, 2) #Pooling reduces it to (grid_length - kernel_size) / 2
        self.conv2 = nn.Conv2d(6, 6, kernel_size) #Again reduces to the size of (grid_length - 3 * kernel_size) / 4
        self.fc1 = nn.Linear(6 * (grid_length - 3 * kernel_size) / 2, 16)
        self.lstm = nn.LSTM(16, hidden_size=hidden_size, num_layers = num_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, obs, hidden):
        x = self.pool(F.relu(self.conv1(obs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x,h = self.lstm(x, hidden)

        return x, h
    
    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
