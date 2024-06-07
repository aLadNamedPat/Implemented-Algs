import gymnasium as gym
from DDPG import DDPG
from nDDPG import nDDPG

def test_model(env, model, episodes=5):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not truncated and not done:
            action = model.test_take_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
        print(f"Episode {ep + 1} completed.")


env = gym.make("BipedalWalker-v3", render_mode = None)
model = nDDPG(env)
# model.load_model()
model.train(3000)
model.save_model()
# model.train(100000)
env_test = gym.make("BipedalWalker-v3", render_mode = "human")
test_model(env_test, model)