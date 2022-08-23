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
from gym_cooking.environment import cooking_zoo
from gym_cooking.cooking_book.recipe_drawer import RECIPES


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
n_agents = 1
num_humans = 0
render = False

level = 'open_room_interaction2'
record = False
max_num_timesteps = 100

goal_encodings = {name: recipe().goal_encoding for name, recipe in RECIPES.items()}

recipes = ["TomatoLettuceSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"])

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]

tom_model = None
impact_model = None

# parameters
num_frames = 1000000
memory_size = 50000
initial_mem_requirement = 5000
batch_size = 512
target_update = 100
obs_dim = obs_space.shape[0]
gamma = 0.99
# PER parameters
alpha = 0.2
n_step = 3

agent = SoftInteractionAgent(tom_model, impact_model, obs_space, action_space, batch_size, target_update,
                             initial_mem_requirement, obs_dim, memory_size, alpha, n_step, gamma)

agent.add_new_goal(tuple(goal_encodings[recipes[0]]))
agent.switch_active_goal(tuple(goal_encodings[recipes[0]]))
agent.is_test = False

# parameters

evalpy_config = {"project_path": "./", "project_folder": "test_logs/", "experiment_name": "cooking_tomato_soft_example"}
agent_dir = "agents/"
idx = 0
agent_save_string = f"soft_agent{idx}_9x9_tomato_salad_test.pickle"
# train
agents = {"player_0": agent}
train(agents, env, num_frames, agent_save_string, agent_dir, checkpoint_save=50000, evalpy_config=evalpy_config)

with open(f'{agent_dir}{agent_save_string}', "wb") as output_file:
    pickle.dump(agent, output_file)
