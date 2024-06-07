from pettingzoo.mpe import simple_adversary_v3
from Network import Agent
from ReplayBuffer import ReplayBuffer
import numpy as np


def train(agent, env, num_ep = 1000, max_env_steps = 128, batch_size = 256):
    c_ep = 0
    tt = 0
    s_hist = []
    while c_ep < num_ep:
        i = 0
        t_score = 0
        obs, info = env.reset()
        while i <= max_env_steps:
            actions = agent.choose_action_wn(obs, continuous = False)
            print(actions)
            next_obs, rewards, done, truncated, info = env.step(actions)
            agent.memory.add_memory(obs, next_obs, actions, rewards, done)

            obs = next_obs

            i += 1
            tt += 1
            if tt % 50 == 0 and agent.memory.ready():
                agent.learn(batch_size)
            
            if evaluate_done():
                break
            t_score += np.sum(rewards)
        s_hist.append(t_score)
    
    return s_hist

def evaluate_done(done):
    return done

def test_model(env, model, episodes=5):
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
            print(dones)
            print(truncateds)
            env.render()
        print(f"Episode {ep + 1} completed.")
    
env = simple_adversary_v3.parallel_env(render_mode = "None", max_cycles = 256, continuous_actions=False)

actions = {agent: env.action_space(agent).sample() for agent in env.reset()[0].keys()}
print(actions)
rp = ReplayBuffer()
model = Agent(3, env, rp, categorical = True)
# model.load_model()
rewards = train(model, env)
print(rewards)
# model.save_model()
# model.train(100000)
env_test = simple_adversary_v3.parallel_env(render_mode = "human", max_cycles = 256, continuous_actions=False)
test_model(env_test, model)
