import sys 
import os
import time
from DQN import DQN
# from DQN_LSTM import DQN

sys.path.append(os.path.abspath("../"))
from NoCommGrid.env.GridEnv import GridEnv

env = GridEnv(1, 1,render_mode= "human", grid_size=3, max_timesteps=64, diff_obs = "v2")


# print(env.observation_space(""))
# running = True
# obs, info = env.reset()
# while running:
#     observations, rewards, terminations, truncations, info = env.sample()
#     time.sleep(5)
#     if any(terminations.values()):
#         break

m = DQN(env, 5, 5)
m.batch_train(100)
m.plot_rewards()
env = GridEnv(1, 1, render_mode="human", grid_size=3, max_timesteps=32, diff_obs='v2')
model = DQN(env, 5, 5)
model.load_models()
model.evaluate(10)

# model = IPPO(env , 1)
# # print("running")
# model.batch_train(500000)
# # plt.plot(model.total_rewards)
# # print(model.total_rewards)
# model.save_models()
# env = GridEnv(1, 4, render_mode="human", grid_size=3, max_timesteps=64, diff_obs = "v2")
# model = IPPO(env , 1)
# model.load_models()
# model.batch_train(100000)
# # model.train(100000)