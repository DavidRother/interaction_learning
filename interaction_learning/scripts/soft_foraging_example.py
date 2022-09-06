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
from interaction_learning.algorithms.soft_interaction_agent import SoftInteractionAgent
from interaction_learning.core.training_ma_env import train
from lbforaging.foraging import zoo_environment


seed = 123463


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# environment
players = 1
max_player_level = 3
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
normalize_reward = True
grid_observation = False
penalty = 0.0

goal_encodings = {"max_points": [1, 0]}

# env = gym.make("Foraging-8x8-2p-2f-v2")
env = zoo_environment.parallel_env(players=players, max_player_level=max_player_level, field_size=field_size,
                                   max_food=max_food, sight=sight, max_episode_steps=max_episode_steps,
                                   force_coop=force_coop, normalize_reward=normalize_reward,
                                   grid_observation=grid_observation, penalty=penalty)

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]

tom_model = None
impact_model = None

# parameters
num_frames = 200000
memory_size = 10000
initial_mem_requirement = 1000
batch_size = 512
target_update = 100
obs_dim = obs_space.shape[0]
gamma = 0.99
# PER parameters
alpha = 0.2
n_step = 3
entropy_alpha = 3.2

agent = SoftInteractionAgent(tom_model, impact_model, obs_space, action_space, batch_size, target_update,
                             initial_mem_requirement, obs_dim, memory_size, alpha, n_step, gamma, entropy_alpha)

agent.add_new_goal(tuple(goal_encodings["max_points"]))
agent.switch_active_goal(tuple(goal_encodings["max_points"]))
agent.is_test = False

# parameters

evalpy_config = {"project_path": "./", "project_folder": "test_logs/", "experiment_name": "cooking_tomato_soft_example"}
agent_dir = "agents/"
idx = 0
agent_save_string = f"soft_agent{idx}_7x7_tomato_salad_test.pickle"
# train
agents = {"player_0": agent}
train(agents, env, num_frames, agent_save_string, agent_dir, checkpoint_save=50000, evalpy_config=evalpy_config)

with open(f'{agent_dir}{agent_save_string}', "wb") as output_file:
    pickle.dump(agent, output_file)
