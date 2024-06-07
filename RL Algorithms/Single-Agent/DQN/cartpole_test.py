import gym
from DQN_Cartpole import DQN

env = gym.make('CartPole-v0', render_mode='rgb_array')

m = DQN(env)
m.batch_train(100)
m.plot_rewards()
m.save_models("DQN_Cartpole.pth")

env = gym.make("CartPole-v0", render_mode='human')
a = DQN(env)
a.load_models()
a.batch_train(100)