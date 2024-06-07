import torch
import random
from collections import deque

class ReplayBuffer:
    #I need to store the past rewards, states, actions, and 
    def __init__(self, max_len = 1e6, batch_size = 128):
        self.memory = deque(maxlen=max_len)
        self.num_stored = 0
        self.sample_ready = False
        self.batch_size = batch_size
        
    def sample(self):
        
        if not self.sample_ready:
            return RuntimeError("Not enough samples stored!")
        sampled_experiences = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp[0] for exp in sampled_experiences])
        rewards = torch.stack([exp[1] for exp in sampled_experiences])
        next_states = torch.stack([exp[2] for exp in sampled_experiences])
        dones = torch.stack([exp[3] for exp in sampled_experiences])

        return states, rewards, next_states, dones

    def add(self, state, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float)
        reward = torch.tensor([reward], dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        done = torch.tensor([done], dtype = torch.float)

        self.memory.append((state, reward, next_state, done))
        if self.num_stored <= len(self.memory):
            self.num_stored += 1

        if self.num_stored >= self.batch_size:
            self.sample_ready = True

    def get_size(self):
        return len(self.memory)