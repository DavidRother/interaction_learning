import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from collections import deque
import random
from torch.distributions import Categorical
import gym
import numpy as np
from interaction_learning.algorithms.soft_dqn.soft_dqn_utils import Memory, SoftQNetwork
from partigames.environment.zoo_env import parallel_env
import matplotlib.pyplot as plt
from time import sleep


device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

agent_position_generator = lambda: [np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])]


agent_reward = [f"y0d", ""]
max_steps = 1000
ghost_agents = 0
render = True
num_agents = 2

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                   agent_reward=agent_reward, max_steps=max_steps, ghost_agents=ghost_agents, render=render)
# temperature = torch.Tensor(alpha)
alpha = 0.1
impact_alpha = 0.3
net_1 = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                     alpha, device="cpu").to(device)
net_2 = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                     impact_alpha, device="cpu").to(device)

net_1.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/agent/sql_final_policy_y0d"))
net_2.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/agent/sql_final_policy_y0d_impact"))
# entropy_optimizer = torch.optim.Adam(temperature, lr=1e-4)

state = env.reset()
episode_reward = 0
for time_steps in range(max_steps):
    # action = net_1.select_action(state["player_0"])
    action = 0
    action_2 = net_2.select_action(state["player_1"])
    actions = {"player_0": action, "player_1": action_2}
    next_state, reward, done, _ = env.step(actions)
    episode_reward += reward["player_0"]
    print(reward)

    env.render(mode="human")
    sleep(0.1)

    if any(done.values()):
        break

    state = next_state

print(episode_reward)

