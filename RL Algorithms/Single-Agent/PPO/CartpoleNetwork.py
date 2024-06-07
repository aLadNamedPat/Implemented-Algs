import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, categorical = False):
        super(ActorNetwork, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        # self.fc1 = nn.Linear(obs_dim, 128)
        # self.dropout = nn.Dropout(0.03)
        # self.fc2 = nn.Linear(128, 128)
        # self.dropout2 = nn.Dropout(0.03)
        # self.actor_head = nn.Linear(128, act_dim)
        self.categorical = categorical
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # x = F.relu(self.dropout(self.fc1(obs)))
        # x = F.relu(self.dropout2(self.fc2(x)))
        if self.categorical:
            x = F.softmax(self.actor_net(obs), dim=-1)
        else: 
            x = F.tanh(self.actor_net(obs))

        return x  # Provides the current action to take at a given state (policy)
    

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(CriticNetwork, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.fc1 = nn.Linear(obs_dim, 128)
        # self.dropout = nn.Dropout(0.02)
        # self.fc2 = nn.Linear(128, 128)
        # self.dropout2 = nn.Dropout(0.02)
        # self.critic_head = nn.Linear(128, 1)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # x = F.relu(self.dropout(self.fc1(obs)))
        # x = F.relu(self.dropout2(self.fc2(x)))
        # x = self.critic_head(x)
        x = self.critic_net(obs)
        return x #Predicts the value of a current state