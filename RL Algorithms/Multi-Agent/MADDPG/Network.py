import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical = False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.categorical = categorical
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        if self.categorical:
            x = torch.softmax(self.fc3(x), dim=0)
        else: 
            x = torch.tanh(self.fc3(x))
        return x  # Provides the current action to take at a given state (policy)
    
    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 2e-3)
    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float)
        
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype = torch.float)

        x = torch.cat([obs, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self, num_agents, env, memory, tau = 0.01, gamma = 0.99, categorical = False):
        self.tau = tau
        self.num_agents = num_agents
        self.gamma = gamma
        self.env = env
        self.agent_keys = env.reset()[0].keys()
        #Replay buffer memory here
        self.memory = memory

        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        for i, agent in enumerate(self.agent_keys):
            print(env.observation_space(agent).shape[0])
            action_dim = env.action_space(agent).n
            obs_dim = env.observation_space(agent).shape[0]
            actor = Actor(obs_dim, action_dim, categorical)
            critic = Critic(obs_dim, action_dim)
            n_target_actor = Actor(obs_dim, action_dim, categorical)
            n_target_critic = Critic(obs_dim, action_dim)

            n_target_actor.load_state_dict(actor.state_dict())
            n_target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(n_target_actor)
            self.target_critics.append(n_target_critic)

    
    def choose_action_wn(self, obs, continuous = True,  lower_bound = -1, upper_bound = 1, std = 0.3):
        actions = {}
        keys = obs.keys()
        for i, agent in enumerate(keys):
            action = self.actors[i](obs[agent])
            if continuous:
                rv = torch.normal(action, std)
                action = torch.clip(action + rv, lower_bound, upper_bound)
            else: 
                action = action.argmax()
                rv = torch.randint(self.env.action_space(agent).n, (1, 1))
                action = int(torch.clip(action + rv.squeeze(), 0, self.env.action_space(agent).n - 1))
            actions[agent] = action
        return actions

    # def choose_action_wn(self, obs, dict_keys, continuous = True,  lower_bound = -1, upper_bound = 1, std = 0.3):
    #     actions = np.zeros(self.num_agents)

    #     for i, agent in enumerate(dict_keys):
    #         actions[i] = self.actors[i](obs[agent])
    #         if continuous:
    #             rv = torch.normal(actions[i], std)
    #             torch.clip(actions[i] + rv, lower_bound, upper_bound)
    #         else: 
    #             rv = torch.randint(0, self.action_dim)
    #             torch.clip(actions[i] + rv, 0, self.action_dim - 1)

    #     return actions

    def choose_action(self, obs):
        actions = []
        keys = obs.keys()
        for i,agent in enumerate(keys):
            actions.append(self.actors[i](obs[agent]))
        return actions
    
    def update_targets(self):
        for i in range(self.num_agents):
            for target_param, org_params in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data_copy_(target_param * (1 - self.tau) + org_params * self.tau)

            for atarget_param, aorg_params in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                atarget_param.data_copy_(target_param * (1 - self.tau) + aorg_params * self.tau)

    #Predict the value of the next point for approximating the returns on rewards
    #Next_batch_obs, batch_rewards, and batch_done should all be tensors
    def compute_targets(self, global_next_batch_obs, next_batch_obs, batch_rewards, batch_done):
        target_means = list(self.num_agents)
        target_q = list(self.num_agents)
        q_targets = list(self.num_agents)
        #save batch_observations as a list of numpy arrays in the memory buffer
        #Batch observations should look something like this: [[batch_obs1], [batch_obs2], [batch_obs3], ...]
        for i in range(self.num_agents):
            target_means[i] = self.target_actors[i](next_batch_obs[i])
        target_means = torch.tensor(target_means, dtype = torch.float)
        for i in range(self.num_agents):
            target_q[i] = self.target_critics[i](global_next_batch_obs, target_means).squeeze()
            q_targets[i] = batch_rewards[i] + self.gamma * target_q[i] * (1 - batch_done[i].long())
        
        return np.array(q_targets)
    
    def compute_q(self, global_batch_obs, global_batch_actions):
        qs = list(self.num_agents)
        for i in range(self.num_agents):
            qs[i] = self.critics[i](global_batch_obs, global_batch_actions)
        return np.array(qs)
    
    def learn(self, batch_size):
        #Centralized training decentralized execution paradigm means that we should use all the observaitons to train the agents to learn the task
        #Need to store all the individual agent actions, individual agent observations, individual agent next observations
        #all the rewards, all the agent dones, all the observations, and all the next observations concatenated together 
        if self.memory.ready():
            batch_obs, batch_actions, batch_new_obs, batch_agent_obs, batch_agent_actions, batch_agent_new_obs, batch_agent_rewards, batch_agent_dones = self.memory.sample(batch_size)
        
        q_targets = self.compute_targets(batch_new_obs, batch_agent_new_obs, batch_agent_rewards, batch_agent_dones)
        qs = self.compute_q(batch_obs, batch_actions)

        for i in range(self.num_agents):
            critic_loss = F.mse_loss(qs, q_targets)
            self.critics[i].learn(critic_loss)

            actor_loss = -self.critics[i](batch_obs, self.choose_action(batch_agent_obs))
            self.actors[i].learn(actor_loss)
        
        self.update_targets()