import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, categorical = False, lr_actor = 1e-3):
        super(ActorNetwork, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(0.02),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.02),
            nn.Linear(128, act_dim)
        ) 
        # self.fc1 = nn.Linear(obs_dim, 512)
        # self.ln1 = nn.LayerNorm(512)
        # self.dp1 = nn.Dropout(0.01)
        # self.fc2 = nn.Linear(512, 256)
        # self.ln2 = nn.LayerNorm(256)
        # self.dp2 = nn.Dropout(0.01)
        # self.fc3 = nn.Linear(256, 64)
        # self.ln3 = nn.LayerNorm(64)
        # self.dp3 = nn.Dropout(0.01)
        # self.actor_head = nn.Linear(64, act_dim)
        self.categorical = categorical
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr_actor)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # x = F.relu(self.ln1(self.fc1(obs)))
        # x = self.dp1(x)
        # x = F.relu(self.ln2(self.fc2(x)))
        # x = self.dp2(x)
        # x = F.relu(self.ln3(self.fc3(x)))
        # x = self.dp3(x)
        if self.categorical:
            x = torch.softmax(self.actor_net(obs), dim=0)
        else: 
            x = torch.tanh(self.actor_net(obs))
        return x  # Provides the current action to take at a given state (policy)

    def train_agent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, lr_critic = 1e-3):
        super(CriticNetwork, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(0.02),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.01),
            nn.Linear(128, 1)
        )
        # self.fc1 = nn.Linear(obs_dim, 512)
        # self.ln1 = nn.LayerNorm(512)
        # self.dp1 = nn.Dropout(0.02)
        # self.fc2 = nn.Linear(512, 256)
        # self.ln2 = nn.LayerNorm(256)
        # self.dp2 = nn.Dropout(0.02)
        # self.fc3 = nn.Linear(256, 64)
        # self.ln3 = nn.LayerNorm(64)
        # self.dp3 = nn.Dropout(0.01)
        # self.critic_head = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr_critic)
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # x = F.relu(self.ln1(self.fc1(obs)))
        # x = self.dp1(x)
        # x = F.relu(self.ln2(self.fc2(x)))
        # x = self.dp2(x)
        # x = F.relu(self.ln3(self.fc3(x)))
        # x = self.dp3(x)
        x = self.critic_net(obs)

        return x #Predicts the value of a current state
    
    def train_agent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
