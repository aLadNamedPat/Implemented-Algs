import torch
from torch.nn import functional as F
from CartpoleNetwork import ActorNetwork, CriticNetwork
from torch.distributions import Normal, MultivariateNormal, Categorical # Assuming continuous actions; adjust if discrete
import numpy as np

class PPO():
    def __init__(self, env):
        self.env = env
        self.__initialize__hyperparams()
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape[0]
        print(self.action_dim)
        self.actor = ActorNetwork(self.obs_dim, self.action_dim, categorical = True)
        self.critic = CriticNetwork(self.obs_dim)

        # self.cov_var = torch.full(size = (self.action_dim,), fill_value = self.stdv)
        # self.cov_mat = torch.diag(self.cov_var)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    # def train(self, total_time_steps):
    #     self.actor.train()
    #     self.critic.train()
    #     t = 0
    #     while t < total_time_steps:
    #         obs, info = self.env.reset()
    #         done = False
    #         for _ in range(self.timesteps_per_episode):
    #             num_steps = _
    #             mean = self.actor(obs) #Actor generates the actual actions for the agent to take
    #             dist = Categorical(mean)  #Sample actions that should be taken
    #             actions = dist.sample()
    #             action_log_prob = dist.log_prob(actions).detach() #Compute the log probability of the probability distributions
    #             # actions_clamped = torch.clamp(actions, -1, 1)

    #             value = self.critic(obs).squeeze()
    #             next_obs, reward, done, truncated, info = self.env.step(action_log_prob)
    #             reward = torch.tensor(reward, dtype=torch.float)  # Ensure reward is a tensor
    #             for i in range(self.num_epochs_per_training):
    #                 #Compute the sampling weight based on the old network and the updated network
    #                 mean = self.actor(obs)
    #                 dist = Categorical(mean)  #Sample actions that should be taken
    #                 # sampling_weight = dist.log_prob(self.actor(obs)).detach() - action_log_prob
    #                 sampling_weight = (dist.log_prob(self.actor(obs)) - action_log_prob).exp()

    #                 # exp_sampling_weight = torch.exp(sampling_weight)
    #                 if done == True or t == self.timesteps_per_episode:
    #                     advantage = reward - value
    #                     critic_target = reward
    #                 else:
    #                     advantage = reward + self.gamma * self.critic(next_obs) - value
    #                     critic_target = reward + self.gamma * self.critic(next_obs)
    #                 # critic_target = critic_target.requires_grad_(True)
    #                 # value = value.requires_grad_(True)
    #                 actor_loss = (-torch.min(sampling_weight * advantage, torch.clamp(sampling_weight, 1 - self.clip, 1 + self.clip) * advantage)).mean()
    #                 critic_loss = torch.nn.MSELoss()(value, critic_target)

    #                 #Update the actor network
    #                 self.actor_optimizer.zero_grad()
    #                 actor_loss.backward(retain_graph=True)
    #                 self.actor_optimizer.step()

    #                 #Update the critic network
    #                 self.critic_optimizer.zero_grad()
    #                 critic_loss.backward(retain_graph=True)
    #                 self.critic_optimizer.step() 
    #             obs = next_obs #Update the current observation to the next observation
    #             if done:
    #                 break

    #         t = t + num_steps #Update the number of timesteps taken in the current iteration of training
    #         print(f"Total timesteps taken: {t}")

    def batch_train(self, total_time_steps):
        self.actor.train()
        self.critic.train()
        t = 0
        # actor_loss_per_round = []
        # critic_loss_per_round = []
        # rtgs_per_round = []
        while t < total_time_steps:
            batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens = self.batch_updates() #Obtain the batch updates
            adv = batch_rtgs - self.critic(batch_obs).squeeze().detach() #Compute the advantage
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) #Normalize the advantage
            print(adv)
            for i in range(self.num_epochs_per_training):
                action_prob = self.actor(batch_obs) #This is the discretized action probability that is generated
                # print(action_prob)
                # print(Val)
                dist = Categorical(action_prob)
                curr_log_prob = dist.log_prob(batch_acts) #Compute the current log probability from the batch_actions that were sampled
                # Val, curr_log_prob = self.evaluate_value(batch_obs)
                p_weight = (curr_log_prob - batch_log_acts).exp()
                print(p_weight)
                clipped_weight = torch.clamp(p_weight, 1 - self.clip, 1 + self.clip)
                comp1 = p_weight * adv
                comp2 = clipped_weight * adv 
                actor_loss = (-torch.min(comp1, comp2)).mean()
                
                Val = self.critic(batch_obs).squeeze()
                critic_loss = F.smooth_l1_loss(batch_rtgs, Val)
                # print("actor loss: ", actor_loss)
                # print("critic loss: ", critic_loss)
                # #Add entropy to the problem
                # dist = Categorical(self.actor(batch_obs))
                # entropy_bonus = dist.entropy().mean().detach()
                # actor_loss = actor_loss - 0.001 * entropy_bonus

                actor_loss = actor_loss.requires_grad_(True)
                critic_loss = critic_loss.requires_grad_(True) 

                print(actor_loss)
                #train the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                #train the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()
                # print("actor loss per round: ", actor_loss)
                # print("critic loss per round: ", critic_loss)

            t += np.sum(batch_lens)
            print(f"total_timesteps: {t}")
            # print("actor_loss over rounds", actor_loss_per_round)
            # print("critic_loss over rounds", critic_loss_per_round)
    
    def __initialize__hyperparams(self):
        self.timesteps_per_episode = 2048
        self.batch_timesteps = 2048
        self.num_epochs_per_training = 5
        self.gamma = 0.99
        self.clip = 0.2
        self.lr_actor = 0.01
        self.lr_critic = 0.01
        self.stdv = 0.05

    def batch_updates(self):
        batch_obs = [] #save the observations
        batch_rews = [] #save the rewards gained per episode
        batch_lens = [] #save the length of the episodes
        batch_acts = [] #save the actions taken
        batch_log_acts = [] #compute the log probabilities of the actions taken
        batch_rtgs = []   #compute the batch returns to go
        t = 0
        while t < self.batch_timesteps:
            episode_reward = []
            obs, info = self.env.reset()
            done = False
            for ep_len in range(self.timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, action_log_prob = self.take_action(obs)

                obs, reward, done, truncated, info = self.env.step(action.numpy())

                batch_acts.append(action)
                episode_reward.append(reward)
                batch_log_acts.append(action_log_prob)

                if done:
                    break
            batch_lens.append(ep_len + 1)
            batch_rews.append(episode_reward)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_acts = torch.tensor(np.array(batch_log_acts), dtype=torch.float)
        batch_rtgs = torch.tensor(self.find_rewards_to_go(batch_rews), dtype = torch.float)
        return batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens
    

    def find_rewards_to_go(self, rewards):
        #Store the returns from each episode at every state
        rtgs = []
        for rew_episodes in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(rew_episodes):
                discounted_reward = reward + self.gamma * discounted_reward
                rtgs.insert(0, discounted_reward) #Place the new discounted reward at the front of the list (since the rewards are reversed)
        return rtgs

    def take_action(self, obs):
        action = self.actor(obs)
        dist = Categorical(action)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).detach()
        return action.squeeze(), action_log_prob
    
    def evaluate_value(self, obs):
        action_prob = self.actor(obs)
        dist = Categorical(action_prob)
        action = dist.sample()
        # log_prob = dist.log_prob(action).detach()
        # value = self.critic(obs).squeeze().detach()
        log_prob = dist.log_prob(action)
        value = self.critic(obs).squeeze()

        return value, log_prob
        
    def save_model(self, filename='ppo_model.pth'):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load_model(self, filename='ppo_model.pth'):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])