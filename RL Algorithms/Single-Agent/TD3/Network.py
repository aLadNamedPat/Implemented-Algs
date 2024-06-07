import torch
import torch.nn as nn
import numpy as np

#Predicts the best action to take
#DDPG is DQN with continuous action space
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # print(f"inside the actor network {action_dim}")
        # state_dim = int(state_dim)
        # print(state_dim)
        # print(action_dim)
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # print(self.actor_net(obs))
        x = self.dropout(self.relu(self.fc1(obs)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x
 
#Generates the q values
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)
        # self.network = nn.Sequential(
        #     nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.02),
        #     nn.Linear(256 + action_dim, 256), nn.ReLU(), nn.Dropout(0.02),
        #     nn.Linear(256, 1)
        # )

    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)

        out = self.dropout(self.relu(self.fc1(obs)))
        out = self.dropout(self.relu(self.fc2(torch.cat([out, action], 1))))
        out = self.fc3(out)
        return out