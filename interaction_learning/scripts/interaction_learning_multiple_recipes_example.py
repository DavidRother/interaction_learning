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
from interaction_learning.algorithms.interaction_agent import InteractionAgent
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

level = 'open_room_interaction'
record = False
max_num_timesteps = 100

goal_encodings = {name: recipe().goal_encoding for name, recipe in RECIPES.items()}

recipes_to_learn = ["TomatoLettuceSalad", "CarrotBanana", "CucumberOnion", "AppleWatermelon"]
recipes = ["TomatoLettuceSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env_phase1 = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                                      recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"],
                                      ghost_agents=1)

obs_space, action_space = env_phase1.observation_spaces["player_0"], env_phase1.action_spaces["player_0"]

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

v_min = 0  # 0
v_max = 200  # 200

agent = InteractionAgent(tom_model, impact_model, obs_space, action_space, batch_size, target_update,
                         initial_mem_requirement, obs_dim, memory_size, alpha, n_step, gamma)


for recipe in recipes_to_learn:

    env_phase1 = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                                          recipes=[recipe], action_scheme=action_scheme, obs_spaces=["feature_vector"],
                                          ghost_agents=1)

    obs_space, action_space = env_phase1.observation_spaces["player_0"], env_phase1.action_spaces["player_0"]

    agent.add_new_goal(tuple(goal_encodings[recipes[0]]))
    agent.switch_active_goal(tuple(goal_encodings[recipes[0]]))
    agent.is_test = False

    # parameters

    evalpy_config = {"project_path": "./", "project_folder": "test_logs/",
                     "experiment_name": f"cooking_{recipe}_example"}
    agent_dir = "agents/"
    idx = 0
    agent_save_string = f"agent{idx}_7x7_interaction_test.pickle"
    # train
    agents = {"player_0": agent}
    train(agents, env_phase1, num_frames, agent_save_string, agent_dir, checkpoint_save=50000, evalpy_config=evalpy_config)

    with open(f'{agent_dir}{agent_save_string}', "wb") as output_file:
        print("saving Interaction Agent")
        pickle.dump(agent, output_file)

print("Finished Single Agent Learning")
# Learn interaction now with each recipe once
agent.interaction_learning = True
agent.interactive_mode = True

for recipe_1, recipe_2 in zip(recipes_to_learn, reversed(recipes_to_learn)):
    other_agent = agent.agents[goal_encodings[recipe_2]]
    agents = {"player_0": agent, "player_1": other_agent}
    n_agents = 2
    ghost_agents = 0

    recipes = [recipe_1, recipe_2]

    env_phase1 = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                                          recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"],
                                          ghost_agents=ghost_agents)

    obs_space, action_space = env_phase1.observation_spaces["player_0"], env_phase1.action_spaces["player_0"]

    evalpy_config = {"project_path": "./", "project_folder": "test_logs/",
                     "experiment_name": f"cooking_{recipes}_example"}
    agent_dir = "agents/"
    idx = 0
    agent_save_string = f"agent{idx}_7x7_interaction_test.pickle"
    # train

    train(agents, env_phase1, num_frames, agent_save_string, agent_dir, checkpoint_save=50000,
          evalpy_config=evalpy_config)
