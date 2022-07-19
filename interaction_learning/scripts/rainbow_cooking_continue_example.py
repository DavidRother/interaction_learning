from typing import Dict, List, Tuple
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pickle

from interaction_learning.algorithms.rainbow_agent import DQNAgent
from interaction_learning.core.training import train
from gym_cooking.environment import cooking_zoo

# environment
n_agents = 1
num_humans = 0
render = False

level = 'open_room_salad3'
seed = 123463
record = False
max_num_timesteps = 100
recipes = ["TomatoSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"])

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 300000
memory_size = 10000
initial_mem_requirement = 5000
batch_size = 256
target_update = 1000

# train
with open(r"agent3_7x7_tomato_salad.pickle", "rb") as output_file:
    agent = pickle.load(output_file)

train(agent, env, num_frames, plotting_interval=50000)

with open(r"agent3_7x7_tomato_salad.pickle", "wb") as output_file:
    pickle.dump(agent, output_file)
