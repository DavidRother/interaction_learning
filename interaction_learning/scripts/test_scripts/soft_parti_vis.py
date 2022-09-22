import pickle
from partigames.environment.zoo_env import parallel_env
from typing import Dict, List, Tuple
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pickle
from time import sleep

seed = 1


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# environment
num_agents = 1
agent_position_generator = lambda: [np.asarray([0.05, np.random.uniform(0.01, 0.99, 1).item()])]
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 0
render = True

# env = gym.make("Foraging-8x8-2p-2f-v2")
env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

goal_encodings = {"max_points": [1, 0]}

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]

with open(r"./agents/soft_agent0_partigame.pickle", "rb") as output_file:
    agent = pickle.load(output_file)

done = {"player_0": False}

obs = env.reset()
env.render()

while not all(done.values()):
    action = agent.select_action(obs["player_0"])
    obs, reward, done, _ = env.step({"player_0": action})
    env.render()
    sleep(0.1)

print("done")
