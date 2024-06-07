import torch
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from OUNoise import OUNoise

class nDDPG:
    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        print(self.action_dim)
        # self.state_dim = int(env.observation_space.shape[0])
        # self.action_dim = int(env.action_space.shape[0])
        # print(self.obs_dim)
        # print(self.action_dim)
        self.initialize_hyperparameters()

        self.buffer = ReplayBuffer(self.buffer_size)
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.critic = Critic(self.obs_dim, self.action_dim)
        
        self.target_actor = Actor(self.obs_dim, self.action_dim)
        self.target_critic = Critic(self.obs_dim, self.action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.fix_parameters(self.target_actor, self.actor)
        # self.fix_parameters(self.target_critic, self.critic)

        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lrActor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lrCritic)
        self.global_steps = 0
        # Initialize SummaryWriter
        self.writer = SummaryWriter()
        self.OUNoise = OUNoise(self.action_dim, 0, 0.15, 0.2)
    

    def take_action(self, obs):
        mean = self.actor(obs)
        mean = torch.tensor(mean)
        # action_to_take = torch.clip(torch.normal(mean, self.std), -1, 1)
        action_to_take = torch.clip(mean + torch.tensor(self.OUNoise.sample(), dtype=torch.float), -1, 1)
        return action_to_take.detach()
    
    def test_take_action(self,obs):
        return torch.tensor(self.actor(obs))
    
    def initialize_hyperparameters(self):
        self.lrActor = 0.001
        self.lrCritic = 0.001
        self.buffer_size = 1000000
        self.batch_size = 64
        self.max_timesteps = 1024
        self.update_constant = 0.001
        self.gamma = 0.99
    # def fix_parameters(self, target, actual):
    #     for target_param, param in zip(target.parameters(), actual.parameters()):
    #         target_param.data.copy_(param.data)

    def update_targets(self, target, actual):
        for target_param, param in zip(target.parameters(), actual.parameters()):
            target_param.data.copy_(self.update_constant * param.data + (1 - self.update_constant) * target_param.data)

    def compute_targets(self, next_batch_obs, batch_reward, batch_done):
        target_mean = self.target_actor(next_batch_obs).detach()
        target_q = self.target_critic(next_batch_obs, target_mean).squeeze()
        return batch_reward + self.gamma * target_q * (1 - batch_done.long())
    
    def update(self):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = self.buffer.sample(self.batch_size)
        batch_rewards = torch.tensor(batch_rewards.clone().detach(), dtype=torch.float)
        target_q = self.compute_targets(batch_next_obs, batch_rewards, batch_done)
        q = self.critic(batch_obs, batch_actions).squeeze()

        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        actor_loss = -self.critic(batch_obs, self.actor(batch_obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # for target_param_actor, param_actor in zip(self.target_actor.parameters(), self.actor.parameters()):
        #     target_param_actor.data.copy_(param_actor.data * self.tau + target_param_actor.data * (1.0 - self.tau))

        self.update_targets(self.target_critic, self.critic)
        self.update_targets(self.target_actor, self.actor)

    def train(self, num_episodes):
        for i in range(num_episodes):
            t = 0
            self.OUNoise.reset()
            obs, info = self.env.reset()
            done = False
            tw = 0
            while not done and t < self.max_timesteps:
                action = self.take_action(obs)
                next_obs, reward, done, info, _ = self.env.step(action)
                self.buffer.add(obs, action, reward, next_obs, done)
                if self.buffer.get_size() > self.batch_size:
                    self.update()
                obs = next_obs
                t += 1
                tw += reward
            if i % 100 == 0:
                self.save_model()
            self.writer.add_scalar("Episode Reward", tw, i)
            self.writer.add_scalar("Episode Length", t, i)
    
    def save_model(self, file_name = "DDPG_model_bipedal.pth"):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.target_actor.state_dict(),
            'critic_target_state_dict': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, file_name)
    
    def load_model(self, filename='DDPG_model_bipedal.pth'):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['actor_target_state_dict'])
        self.target_critic.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])