import gymnasium as gym
from TD3 import TD3

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
model = TD3(env)
model.load_model()
model.train(5000, 10)
model.save_model()
# model.train(100000)
env_test = gym.make("BipedalWalker-v3", render_mode = "human")
test_model(env_test, model)