import torch
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from OUNoise import OUNoise

class TD3:
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
        self.criticOne = Critic(self.obs_dim, self.action_dim)
        self.criticTwo = Critic(self.obs_dim, self.action_dim)

        self.target_actor = Actor(self.obs_dim, self.action_dim)
        self.target_criticOne = Critic(self.obs_dim, self.action_dim)

        self.target_criticTwo = Critic(self.obs_dim, self.action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_criticOne.load_state_dict(self.criticOne.state_dict())
        self.target_criticTwo.load_state_dict(self.criticTwo.state_dict())
        # self.fix_parameters(self.target_actor, self.actor)
        # self.fix_parameters(self.target_critic, self.critic)

        self.target_actor.requires_grad_(False)
        self.target_criticOne.requires_grad_(False)
        self.target_criticTwo.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lrActor)
        self.critic_optimizer_one = torch.optim.Adam(self.criticOne.parameters(), lr=self.lrCritic)
        self.critic_optimizer_two = torch.optim.Adam(self.criticTwo.parameters(), lr=self.lrCritic)
        self.global_steps = 0
        # Initialize SummaryWriter
        self.writer = SummaryWriter()
        self.OUNoise = OUNoise(self.action_dim, 0, 0.15, 0.2)
        self.timesteps_since_last_update = 0
    

    def take_action(self, obs):
        mean = self.actor(obs)
        mean = torch.tensor(mean)
        # action_to_take = torch.clip(torch.normal(mean, self.std), -1, 1)
        action_to_take = torch.clip(mean + torch.clip(torch.tensor(self.OUNoise.sample(), dtype=torch.float), -0.3, 0.3), -1, 1)
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
        target_q_one = self.target_criticOne(next_batch_obs, target_mean).squeeze()
        target_q_two = self.target_criticTwo(next_batch_obs, target_mean).squeeze()
        target_q = torch.min(target_q_one, target_q_two)
        return batch_reward + self.gamma * target_q * (1 - batch_done.long())
    
    def update(self, step):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = self.buffer.sample(self.batch_size)
        batch_rewards = torch.tensor(batch_rewards.clone().detach(), dtype=torch.float)
        target_q = self.compute_targets(batch_next_obs, batch_rewards, batch_done)
        qOne = self.criticOne(batch_obs, batch_actions).squeeze()
        qTwo = self.criticTwo(batch_obs, batch_actions).squeeze()


        critic_loss_one = F.mse_loss(qOne, target_q)
        self.critic_optimizer_one.zero_grad()
        critic_loss_one.backward()
        self.critic_optimizer_one.step()

        critic_loss_two = F.mse_loss(qTwo, target_q)
        self.critic_optimizer_two.zero_grad()
        critic_loss_two.backward()
        self.critic_optimizer_two.step()    
        
        if step % 2 == 0:
            actor_loss = -self.criticOne(batch_obs, self.actor(batch_obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        # for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # for target_param_actor, param_actor in zip(self.target_actor.parameters(), self.actor.parameters()):
        #     target_param_actor.data.copy_(param_actor.data * self.tau + target_param_actor.data * (1.0 - self.tau))

        self.update_targets(self.target_actor, self.actor)
        self.update_targets(self.target_criticOne, self.criticOne)
        self.update_targets(self.target_criticTwo, self.criticTwo)

    def train(self, num_episodes, timesteps):
        for i in range(num_episodes):
            t = 0
            self.OUNoise.reset()
            obs, info = self.env.reset()
            done = False
            tw = 0
            self.timesteps_since_last_update = 0
            while not done and t < self.max_timesteps:
                self.timesteps_since_last_update += 1
                action = self.take_action(obs)
                next_obs, reward, done, info, _ = self.env.step(action)
                self.buffer.add(obs, action, reward, next_obs, done)
                if self.buffer.get_size() > self.batch_size and self.timesteps_since_last_update > timesteps:
                    self.update(t)
                    self.timesteps_since_last_update = 0
                obs = next_obs
                t += 1
                tw += reward
            if i % 100 == 0:
                self.save_model()
            self.writer.add_scalar("Episode Reward", tw, i)
            self.writer.add_scalar("Episode Length", t, i)
    
    def save_model(self, file_name = "TD3_model_bipedal.pth"):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_one_state_dict': self.criticOne.state_dict(),
            'critic_two_state_dict': self.criticTwo.state_dict(),
            'actor_target_state_dict': self.target_actor.state_dict(),
            'critic_target_one_state_dict': self.target_criticOne.state_dict(),
            'critic_target_two_state_dict': self.target_criticTwo.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer_one': self.critic_optimizer_one.state_dict(),
            'critic_optimizer_two': self.critic_optimizer_two.state_dict(),
        }, file_name)
    
    def load_model(self, filename='TD3_model_bipedal.pth'):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.criticOne.load_state_dict(checkpoint['critic_one_state_dict'])
        self.criticTwo.load_state_dict(checkpoint['critic_two_state_dict'])
        self.target_actor.load_state_dict(checkpoint['actor_target_state_dict'])
        self.target_criticOne.load_state_dict(checkpoint['critic_one_target_state_dict'])
        self.target_criticTwo.load_state_dict(checkpoint['critic_two_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer_one.load_state_dict(checkpoint['critic_optimizer_one'])
        self.critic_optimizer_two.load_state_dict(checkpoint['critic_optimizer_two'])