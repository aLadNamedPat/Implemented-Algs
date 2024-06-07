import gymnasium as gym
from PPO_cartpole import PPO

def test_model(env, model, episodes=5):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not truncated and not done:
            action, _ = model.take_action(obs)
            obs, reward, done, truncated, _ = env.step(action.numpy())
            env.render()
        print(f"Episode {ep + 1} completed.")

env = gym.make("LunarLander-v2", render_mode = None)
model = PPO(env)
# model.load_model()
model.batch_train(200000)
# model.save_model()
# model.train(2000)
env_test = gym.make("LunarLander-v2", render_mode = "human")
test_model(env_test, model)