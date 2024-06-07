import numpy as np
import random 
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, max_size = 100000):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.ready_to_run = False

    def sample(self, batch_size):
        if self.ready_to_run or self.ready(batch_size):
            self.ready_to_run = True
        else:
            return BufferError()
        sampled_experiences = random.sample(self.memory, batch_size)
        batch_g_states = torch.stack([exp[0] for exp in sampled_experiences])
        batch_g_acts = torch.stack([exp[1] for exp in sampled_experiences])
        batch_g_nstates = torch.stack([exp[2] for exp in sampled_experiences])
        batch_states = torch.stack([exp[3] for exp in sampled_experiences])
        batch_actions = torch.stack([exp[4] for exp in sampled_experiences])
        batch_nstates = torch.stack([exp[5] for exp in sampled_experiences])
        batch_rewards = torch.stack([exp[6] for exp in sampled_experiences])
        batch_dones = torch.stack([exp[7] for exp in sampled_experiences])

        return batch_g_states, batch_g_acts, batch_g_nstates, batch_states, batch_actions, batch_nstates, batch_rewards, batch_dones
    
    def ready(self, batch_size):
        if len(self.memory) >= batch_size:
            self.ready_to_run = True
            return True
        return False
    
    #Observations, actions, rewards, dones should all be passed as:
    #Observations {agent1: [], agent2: [], agent3: []}

    #Information that I need to store in the replay buffer:
    #Global observations, global actions, local observations, local actions, local rewards, local dones
    def add_memory(self, observations, new_observations, actions, rewards, dones):
        to_append = []
        keys = observations.keys()
        
        global_obs = []
        global_acts = []
        global_new_obs = []
        agent_mems_obs = []
        agent_mems_new_obs = []
        agent_mems_acts = []
        agent_mems_rewards = []
        agent_mems_dones = []
        
        for agent in keys:
            global_obs += list(observations[agent])
            global_acts += [actions[agent]]
            global_new_obs += list(new_observations[agent])
            agent_mems_obs.append(list(observations[agent]))
            agent_mems_new_obs.append(list(new_observations[agent]))
            agent_mems_acts.append([actions[agent]])
            agent_mems_rewards.append([rewards[agent]])
            agent_mems_dones.append([dones[agent]])
            
        global_obs = torch.tensor(global_obs, dtype = torch.float)
        global_acts = torch.tensor(global_acts, dtype = torch.float)
        global_new_obs = torch.tensor(global_new_obs, dtype = torch.float)
        agent_mems_obs = torch.tensor(agent_mems_obs, dtype = torch.float)
        agent_mems_new_obs = torch.tensor(agent_mems_new_obs, dtype = torch.float)
        agent_mems_acts = torch.tensor(agent_mems_acts, dtype = torch.float)
        agent_mems_rewards = torch.tensor(agent_mems_rewards, dtype = torch.float)
        agent_mems_dones = torch.tensor(agent_mems_dones, dtype = torch.float)

        to_append.append((global_obs,  global_new_obs, global_acts, agent_mems_obs, agent_mems_new_obs, agent_mems_acts, agent_mems_rewards, agent_mems_dones))