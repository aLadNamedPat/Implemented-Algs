from pettingzoo.mpe import simple_adversary_v3
from PPO import IPPO
import gymnasium as gym

def test_model(env, model, episodes=5):
    print(env)
    for ep in range(episodes):
        obs, info = env.reset()
        keys = obs.keys()
        done = False
        truncated = False
        while not truncated and not done:
            action, _ = model.take_action(obs)
            obs, reward, dones, truncateds, _ = env.step(action)
            for i, agent in enumerate(keys):
                if dones[agent] or truncateds[agent]:
                    done = True
            # print(dones)
            # print(truncateds)
            env.render()
        print(f"Episode {ep + 1} completed.")


env = simple_adversary_v3.parallel_env(render_mode = "None", max_cycles = 1024, continuous_actions=False)
model = IPPO(env , 2, 1, True)
model.load_models()
model.batch_train(300000)
model.save_models()
# model.train(100000)
env_test = simple_adversary_v3.parallel_env(render_mode = "human", max_cycles = 1024, continuous_actions=False)
test_model(env_test, model)