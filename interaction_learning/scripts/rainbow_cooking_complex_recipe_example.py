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

level = 'open_room_salad'
seed = 123463
record = False
max_num_timesteps = 1000
recipes = ["TomatoLettuceOnionSalad"]
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
num_frames = 500000
memory_size = 20000
initial_mem_requirement = 5000
batch_size = 512
target_update = 100

v_min = 0  # 0
v_max = 200  # 200


evalpy_config = {"project_path": "./", "project_folder": "test_logs/", "experiment_name": "cooking_tomato_example"}
agent_dir = "agents/"

for idx in range(1):
    agent_save_string = f"agent{idx}_7x7_tomato_onion_lettuce__salad_test.pickle"
    # train
    agent = DQNAgent(obs_space, action_space, memory_size, batch_size, target_update, initial_mem_requirement,
                     v_min=v_min, v_max=v_max, n_step=3)
    train(agent, env, num_frames, agent_save_string, agent_dir, checkpoint_save=50000, evalpy_config=evalpy_config)

    with open(f'{agent_dir}{agent_save_string}', "wb") as output_file:
        pickle.dump(agent, output_file)
