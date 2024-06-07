import random
import torch
from collections import deque
import copy
class ReplayBuffer():
    def __init__(self, max_size=1e5):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float)
        self.buffer.append(
            (torch.tensor(state, dtype=torch.float),
            action,
            reward,
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(done)))
    
    def get_size(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        sampled_experiences = random.sample(self.buffer, batch_size)
        batch_states = torch.stack([exp[0] for exp in sampled_experiences])
        batch_actions = torch.stack([exp[1] for exp in sampled_experiences])
        batch_rewards = torch.stack([exp[2] for exp in sampled_experiences])
        batch_next_states = torch.stack([exp[3] for exp in sampled_experiences])
        batch_dones = torch.stack([exp[4] for exp in sampled_experiences])
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
