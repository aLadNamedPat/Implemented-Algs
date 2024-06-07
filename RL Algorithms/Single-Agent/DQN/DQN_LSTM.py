import torch
import numpy as np
import random
# from Network import Q_Net
from Network import Q_net_LSTM
from memory import ReplayBuffer
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, env, obs_dim, output_dim, per = False):
        self.__init_hyperparameters__()
        self.env = env
        self.obs_dim = obs_dim
        self.num_actions = output_dim
        self.per = per

        self.memory = ReplayBuffer(self.memory_length, self.batch_size)

        self.model = Q_net_LSTM(self.obs_dim, self.num_actions)

        self.target_model = Q_net_LSTM(self.obs_dim, self.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.rewards_per_episode = []  # List to store rewards for each episode

    def __init_hyperparameters__(self):
        self.num_training_steps = 0
        self.rollouts_per_batch = 10
        self.max_timesteps = 500
        self.min_epsilon = 0.1
        self.max_epsilon = 0.5
        self.memory_length = 100000
        self.batch_size = 256
        self.update_constant = 0.05
        self.timesteps_per_target_update = 1
        self.steps_per_save = 10
        self.gamma = 0.99
        self.num_updates = 3

    def evaluate(self, num_steps = 100):
        for i in range(num_steps):
            self.rollout(False)
            
    def batch_train(self, training_steps = 100):
        # self.model.train()
        self.training_steps = training_steps
        for i in range(training_steps):
            self.rollout()
            self.num_training_steps += 1
            for i in range(self.num_updates):
                if not self.memory.sample_ready:
                    continue
                observations, rewards, next_observations, dones = self.memory.sample()
                for i in range(len(observations)):
                    estimated_q_value = torch.max(self.model(observations[i]))
                    if dones[i]:
                        target_q_value = rewards[i]
                    else:
                        target_q_value = rewards[i] + self.gamma * self.target_model(next_observations[i])[torch.argmax(self.model(next_observations[i]))]
                    # print("estimated",estimated_q_value)
                    # print(target_q_value)
                    loss = F.mse_loss(estimated_q_value, target_q_value[0])
                    self.model.learn(loss)
                # estimated_q_value = torch.max(self.model(observations), dim = 1)[0]
                # # print(estimated_q_value)
                # # print(torch.argmax(self.model(next_observations), dim = 1))
                # # print(self.target_model(next_observations)[torch.argmax(self.model(next_observations), dim = 1)])
                # targetVals = self.target_model(next_observations)
                # # print(targetVals[range(targetVals.shape[0]), torch.argmax(self.model(next_observations), dim = 1)])
                # target_q = rewards + self.gamma * targetVals[range(targetVals.shape[0]), torch.argmax(self.model(next_observations), dim = 1)] * (1 - dones.float())
                # loss = F.mse_loss(estimated_q_value, target_q)
                # self.model.learn(loss)
                    if self.training_steps % 10 == 0:
                        self.update_target()
            print(self.num_training_steps)
            self.save_models()


    #Conducts the sampling used to fill the replay buffer
    def rollout(self, training = True):
        for i in range(self.rollouts_per_batch):
            t = 0
            ep_length = 0
            cumulative_reward = 0
            done = False
            truncated = False
            obs, info = self.env.reset()
            mask = self.extract_mask(obs)
            obs = self.convert_observation(obs["agent0"])
            val = False
            run = False
            while(not done and not truncated and t < self.max_timesteps):
                if random.random() < self.epsilon_decay(training):
                    while (not val):
                        action = int(self.num_actions * random.random())
                        if (mask[action] == 1):
                            val = True
                else:
                    if not run:
                        print(type(obs))
                        actions, hidden = self.model(obs.unsqueeze(0), mask)
                        run = True
                    actions, hidden = self.model(obs.unsqueeze(0), mask, hidden)
                    action = np.argmax(actions.detach().numpy())

                action = {"agent0": action}
                # print(mask)
                next_obs, reward, done, truncated, __ = self.env.step(action)
                mask = self.extract_mask(next_obs)
                next_obs = self.convert_observation(next_obs["agent0"])
                reward = reward["agent0"]
                done = done["agent0"]
                truncated = truncated["agent0"]
                self.memory.add(obs, reward, next_obs, done)
                obs = next_obs
                cumulative_reward += reward
                ep_length += 1
                t += 1
                val = False
            # print(ep_length)
            # print(cumulative_reward)
            self.rewards_per_episode.append(cumulative_reward)

    def extract_mask(self, obs):
        return obs["agent0"]["agent_mask"]
    
    def convert_observation(self, obs):
        obs = obs["observation"]
        n_obs = []
        for i in range(len(obs)):
            if type(obs[i]) is int:
                n_obs.append(obs[i])
            else:
                for j in range(len(obs[i])):
                    n_obs.append(obs[i][j])
        return torch.tensor(n_obs, dtype = torch.float)
    
    def update_target(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.update_constant * param.data + (1 - self.update_constant) * target_param.data)

    def epsilon_decay(self, running = True):
        if running:
            return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (self.training_steps - self.num_training_steps) / self.training_steps
        else:
            return 0
    def save_models(self, filename='DQN.pth'):
        dt = {}
        dt["model"] = self.model.state_dict()
        dt["model_optim"] = self.model.optimizer.state_dict()
        dt["target_model"] = self.target_model.state_dict()
        torch.save(dt, filename)
        # for i in range(self.num_agents):
        #     dt[f'actor_state_dict{i}'] = self.actors[i].state_dict()
        #     dt[f'critic_state_dict{i}'] = self.critics[i].state_dict()
        #     dt[f"actor_optimizer_dict{i}"] = self.actors[i].optimizer.state_dict()
        #     dt[f"critic_optimizer_dict{i}"] = self.critics[i].optimizer.state_dict()

    def load_models(self, filename='DQN.pth'):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model"])
        self.model.optimizer.load_state_dict(checkpoint["model_optim"])
        self.target_model.load_state_dict(checkpoint["target_model"])

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_per_episode, label='Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards per Episode Over Training')
        plt.legend()
        plt.grid(True)
        plt.show()
        # print(checkpoint.keys())
        # for i in range(self.num_agents):
        #     self.actors[i].load_state_dict(checkpoint[f'actor_state_dict{i}'])
        #     self.critics[i].load_state_dict(checkpoint[f'critic_state_dict{i}'])
        #     self.actors[i].optimizer.load_state_dict(checkpoint[f'actor_optimizer_dict{i}'])
        #     self.critics[i].optimizer.load_state_dict(checkpoint[f'critic_optimizer_dict{i}'])