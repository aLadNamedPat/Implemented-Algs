# import torch
# import torch.nn as nn


# class MADDPG:
#     def __init__(self, env):
#         state_dim = env.observation_space[0].shape[0]
#         action_dim = env.action_space[0].n

#         self.actors = []
#         self.critics = []

#         self.target_critics = []

#         #load each actor and critic into a list of actors and critics as shown above
    
#     def choose_action(self, obs):
#         #choose action based on actor
#         actions = []

#         for agent_idx, agent in enumerate(self.actors):
#             action = agent(torch.tensor(obs[agent_idx], dtype=torch.float32))
#             actions.append(action)
        
#         return actions
    
#     def learn(self, memory):
#         if not memory.ready():
#             return 
        
#         actor_states, states, actions, rewards, actor_new_states, new_states, dones = memory.sample()
#         #actor_states
#         #

#         states = torch.tensor(states, dtype=torch.float32) # State stores all the actual states
#         actions = torch.tensor(actions, dtype=torch.float32) # Actions stores all the actual actions
#         rewards = torch.tensor(rewards, dtype=torch.float32) # Rewards store all the rewards received by the agents