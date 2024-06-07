import torch
from torch.nn import functional as F
from Network import ActorNetwork, CriticNetwork
from torch.distributions import Normal, MultivariateNormal, Categorical # Assuming continuous actions; adjust if discrete
import numpy as np
import wandb

class IPPO:
    def __init__(self, env, num_good_agents, num_bad_agents, centralized = False):
        self.env = env
        self.__initialize__hyperparams()
        self.centralized = centralized
        # self.obs_dim = env.observation_space.shape[0]
        self.num_agents = num_good_agents + num_bad_agents
        self.agent_keys = env.reset()[0].keys()
        if not centralized:
            self.actors = []
            self.critics = []

            self.num_agents = num_good_agents + num_bad_agents
            for agent in self.agent_keys:
                self.action_dim = env.action_space(agent).n
                self.obs_dim = env.observation_spaces.get(agent).shape[0]
                print(self.obs_dim)
                self.actors.append(ActorNetwork(self.obs_dim, self.action_dim, categorical=True))
                self.critics.append(CriticNetwork(self.obs_dim))
        
        if centralized:
            self.actors = []
            self.critics = []
            self.cent_obs_dim = 0
            for agent in env.reset()[0].keys():
                self.action_dim = env.action_space(agent).n
                self.obs_dim = env.observation_spaces.get(agent).shape[0]
                if "agent" in agent:
                    self.cent_obs_dim += self.obs_dim
                self.actors.append(ActorNetwork(self.obs_dim, self.action_dim, categorical = True))
            
            for agent in env.reset()[0].keys():
                self.obs_dim = env.observation_spaces.get(agent).shape[0]
                if "agent" in agent:
                    # print(agent)
                    # print(self.cent_obs_dim)
                    self.critics.append(CriticNetwork(self.cent_obs_dim))
                else:
                    print(self.obs_dim)
                    self.critics.append(CriticNetwork(self.obs_dim))

    def batch_train(self, total_time_steps):
        for i in range(self.num_agents):
            self.actors[i].train()
            self.critics[i].train()
        t = 0
        # actor_loss_per_round = []
        # critic_loss_per_round = []
        # rtgs_per_round = []
        while t < total_time_steps:
            batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens, local_obs = self.batch_updates() #Obtain the batch updates
            advs = []
            for i in range(self.num_agents):
                adv = batch_rtgs[i] - self.critics[i](batch_obs[i]).squeeze().detach() #Compute the advantage for each agent
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) #Normalize the advantage
                advs.append(adv)

            for i in range(self.num_epochs_per_training):
                for j in range(self.num_agents):
                    action_prob = self.actors[j](local_obs[j]) #This is the discretized action probability that is generated
                    dist = Categorical(action_prob)
                    curr_log_prob = dist.log_prob(batch_acts[j]) #Compute the current log probability from the batch_actions that were sampled
                    p_weight = (curr_log_prob - batch_log_acts[j]).exp()
                    clipped_weight = torch.clamp(p_weight, 1 - self.clip, 1 + self.clip)
                    comp1 = p_weight * advs[j]
                    comp2 = clipped_weight * advs[j]
                    actor_loss = (-torch.min(comp1, comp2)).mean()
                
                    Val = self.critics[j](batch_obs[j]).squeeze()
                    critic_loss = F.smooth_l1_loss(batch_rtgs[j], Val)

                    actor_loss = actor_loss.requires_grad_(True)
                    critic_loss = critic_loss.requires_grad_(True) 
                    self.actors[j].train_agent(actor_loss)
                    self.critics[j].train_agent(critic_loss)
                    # self.actor_optimizer.zero_grad()
                    # actor_loss.backward(retain_graph=True)
                    # self.actor_optimizer.step()
                    # #train the critic
                    # self.critic_optimizer.zero_grad()
                    # critic_loss.backward(retain_graph=True)
                    # self.critic_optimizer.step()
                    # # print("actor loss per round: ", actor_loss)
                    # # print("critic loss per round: ", critic_loss)

            t += np.sum(batch_lens)
            print(f"total_timesteps: {t}")
            # print("actor_loss over rounds", actor_loss_per_round)
            # print("critic_loss over rounds", critic_loss_per_round)
    
    def __initialize__hyperparams(self):
        self.timesteps_per_episode = 128
        self.batch_timesteps = 256
        self.num_epochs_per_training = 5
        self.gamma = 0.99
        self.clip = 0.2
        self.lr_actor = 0.01
        self.lr_critic = 0.01
        self.stdv = 0.05
        wandb.init(
            project = "IPPO",
            config = {
                "learning_rate" : self.lr_actor,
                "architecture" : "DNN",
                "dataset" :  "reinforcement learning" 
            }
        )
    def batch_updates(self):
        batch_obs = [[] for i in range(self.num_agents)] #save the observations
        local_obs = [[] for i in range(self.num_agents)]
        batch_rews = [] #save the rewards gained per episode
        batch_lens = [] #save the length of the episodes
        batch_acts = [[] for i in range(self.num_agents)] #save the actions taken
        batch_log_acts = [[] for i in range(self.num_agents)] #compute the log probabilities of the actions taken
        batch_rtgs = [[] for i in range(self.num_agents)]   #compute the batch returns to go
        t = 0
        while t < self.batch_timesteps:
            episode_reward = [[] for i in range(self.num_agents)]
            obs, info = self.env.reset()
            done = False
            for ep_len in range(self.timesteps_per_episode):
                t += 1
                if not self.centralized:
                    for i, agent in enumerate(self.agent_keys):
                        batch_obs[i].append(torch.tensor(obs[agent], dtype = torch.float))
                else:
                    agg = np.array([])
                    for i, agent in enumerate(self.agent_keys):
                        if "agent" in agent:
                            agg = np.concatenate([agg, obs[agent]])
                            # print(agg)
                            # agg += obs[agent]
                        else:
                            continue
                    for i, agent in enumerate(self.agent_keys):
                        local_obs[i].append(torch.tensor(obs[agent], dtype = torch.float))

                        if "agent" in agent:
                            batch_obs[i].append(torch.tensor(agg,dtype = torch.float))
                        else:
                            batch_obs[i].append(torch.tensor(obs[agent], dtype = torch.float))

                actions, action_log_probs = self.take_action(obs)
                obs, rewards, dones, truncated, info = self.env.step(actions) #Need to make the actual action numpy not the entire list

                done = True
                for i , agent in enumerate(self.agent_keys):
                    batch_acts[i].append(actions[agent])
                    episode_reward[i].append(rewards[agent])
                    batch_log_acts[i].append(action_log_probs[i])
                    if not done or dones[agent]:
                        done = False

                if done:
                    break
            batch_lens.append(ep_len + 1)
            batch_rews.append(episode_reward)
        # print(batch_obs)
        # batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        for i in range(self.num_agents):
            batch_obs[i] = torch.tensor(np.array(batch_obs[i]), dtype = torch.float)
            local_obs[i] = torch.tensor(np.array(local_obs[i]), dtype = torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_acts = torch.tensor(np.array(batch_log_acts), dtype=torch.float)
        batch_rtgs = torch.tensor(self.find_rewards_to_go(batch_rews), dtype = torch.float)
        return batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens, local_obs
    

    def find_rewards_to_go(self, rewards):
        #Store the returns from each episode at every state

        #Typically this is computed as the rewards to go for each episode
        #Original Structure: batch_rews = [[ep_rews], [ep_rews], [ep_rews]]
        #New Structure: batch_rews = [[[ep_rew1], [ep_rew2], ...], [[ep_rew1], [ep_rew2], ...], ... ]

        #rtgs original structure [0, 1, 3, 4, ...]
        #rtgs new structure [[4, 3, 1, 0], [4, 3, 1, 0]]]
        rtgs = []
        for i in range(self.num_agents):
            rtgs_i = []
            for rew_episodes in reversed(rewards):
                discounted_reward = 0
                for reward in reversed(rew_episodes[i]):
                    discounted_reward = reward + self.gamma * discounted_reward
                    rtgs_i.insert(0, discounted_reward) #Place the new discounted reward at the front of the list (since the rewards are reversed)
                # print(discounted_reward)
            rtgs.append(rtgs_i)
        return rtgs

    def take_action(self, obs):
        actions = {}
        actions_log_prob = np.zeros(self.num_agents)
        for i, agent in enumerate(self.agent_keys):
            action = self.actors[i](obs[agent])
            dist = Categorical(action)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).detach()
            actions[agent] = action.squeeze().numpy()
            actions_log_prob[i] = action_log_prob
        return actions, actions_log_prob
    
    # def evaluate_value(self, obs):
    #     action_prob = self.actor(obs)
    #     dist = Categorical(action_prob)
    #     action = dist.sample()
    #     # log_prob = dist.log_prob(action).detach()
    #     # value = self.critic(obs).squeeze().detach()
    #     log_prob = dist.log_prob(action)
    #     value = self.critic(obs).squeeze()

    #     return value, log_prob
        
    def save_models(self, filename='ppo_model.pth'):
        dt = {}
        for i in range(self.num_agents):
            dt[f'actor_state_dict{i}'] = self.actors[i].state_dict()
            dt[f'critic_state_dict{i}'] = self.critics[i].state_dict()
            dt[f"actor_optimizer_dict{i}"] = self.actors[i].optimizer.state_dict()
            dt[f"critic_optimizer_dict{i}"] = self.critics[i].optimizer.state_dict()

        torch.save(dt, filename)

    def load_models(self, filename='ppo_model.pth'):
        checkpoint = torch.load(filename)
        print(checkpoint.keys())
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint[f'actor_state_dict{i}'])
            self.critics[i].load_state_dict(checkpoint[f'critic_state_dict{i}'])
            self.actors[i].optimizer.load_state_dict(checkpoint[f'actor_optimizer_dict{i}'])
            self.critics[i].optimizer.load_state_dict(checkpoint[f'critic_optimizer_dict{i}'])