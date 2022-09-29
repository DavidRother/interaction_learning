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
from partigames.environment.gym_env import GymPartiEnvironment
import matplotlib.pyplot as plt
from time import sleep


device = torch.device("cpu")

agent_position_generator = lambda: [np.asarray([0.05, np.random.uniform(0, 1)])]
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 1
render = True

alpha = 0.05

env = GymPartiEnvironment(agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                          max_steps=max_steps, ghost_agents=ghost_agents, render=render)
onlineQNetwork = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, alpha, device="cpu").to(device)

onlineQNetwork.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/sql490policy"))

action_distribution = {n: 0 for n in range(env.action_space.n)}

episode_rewards = []

for epoch in range(1):
    state = env.reset()
    episode_reward = 0
    for time_steps in range(max_steps):
        action = onlineQNetwork.select_action(state, greedy=False)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        action_distribution[action] += 1

        env.render(mode="human")
        sleep(0.1)

        if done:
            break

    print(episode_reward)

