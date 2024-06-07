import gymnasium as gym
from PPO import PPO

def test_model(env, model, episodes=5):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not truncated and not done:
            action, _ = model.take_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
        print(f"Episode {ep + 1} completed.")

env = gym.make("BipedalWalker-v3", render_mode = None)
model = PPO(env)
model.load_model()
model.batch_train(300000)
model.save_model()
# model.train(100000)
env_test = gym.make("BipedalWalker-v3", render_mode = "human")
test_model(env_test, model)