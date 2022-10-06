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


agent_reward = [f"y0d", "x"]
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
net_2 = SoftQNetwork(env.observation_spaces["player_0"].shape[0] * 2, env.action_spaces["player_0"].n,
                     impact_alpha, device="cpu").to(device)
net_3 = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                     alpha, device="cpu").to(device)

net_1.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/agent/sql_final_policy_y0d"))
net_2.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/agent/sql_final_policy_y0d_impact"))
net_3.load_state_dict(torch.load("/hri/localdisk/drother/PycharmProjects/interaction_learning/interaction_learning/scripts/soft_q_tests/agent/sql490policy"))


def select_combined_action(x):
    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        q_task = net_3(x)
        value_task = net_3.get_value(q_task)
        q_impact = net_2(x)
        value_impact = net_2.get_value(q_impact)

        q_combined = (q_task + q_impact) / 2
        value_combined = (value_task + value_impact) / 2

        dist = torch.exp((q_combined - value_combined) / alpha)

        c = Categorical(dist)
        a = c.sample()

    return a.item()
# entropy_optimizer = torch.optim.Adam(temperature, lr=1e-4)


episode_rewards_combined = []
episode_rewards_combined_p1 = []
episode_rewards_combined_p0 = []
episode_rewards_single = []
episode_rewards_single_p1 = []
episode_rewards_single_p0 = []

num_trials = 1000

agent_positions = [agent_position_generator() for _ in range(num_trials)]

for idx in range(num_trials):
    state = env.reset()
    episode_reward = 0
    ep_p1_reward = 0
    ep_p0_reward = 0
    for time_steps in range(max_steps):
        action = net_1.select_action(state["player_0"])
        # action = 0
        action_2 = select_combined_action(state["player_1"])
        actions = {"player_0": action, "player_1": action_2}
        next_state, reward, done, _ = env.step(actions)
        episode_reward += reward["player_0"] + reward["player_1"]
        ep_p1_reward += reward["player_1"]
        ep_p0_reward += reward["player_0"]
        # print(reward)

        # env.render(mode="human")
        # sleep(0.1)

        if any(done.values()):
            break

        state = next_state
    episode_rewards_combined.append(episode_reward)
    episode_rewards_combined_p0.append(ep_p0_reward)
    episode_rewards_combined_p1.append(ep_p1_reward)

print("halftime")

for idx in range(num_trials):
    state = env.reset()
    episode_reward = 0
    ep_p1_reward = 0
    ep_p0_reward = 0
    for time_steps in range(max_steps):
        action = net_1.select_action(state["player_0"])
        # action = 0
        action_2 = net_3.select_action(state["player_1"])
        actions = {"player_0": action, "player_1": action_2}
        next_state, reward, done, _ = env.step(actions)
        episode_reward += reward["player_0"] + reward["player_1"]
        ep_p1_reward += reward["player_1"]
        ep_p0_reward += reward["player_0"]
        # print(reward)

        # env.render(mode="human")
        # sleep(0.1)

        if any(done.values()):
            break

        state = next_state
    episode_rewards_single.append(episode_reward)
    episode_rewards_single_p0.append(ep_p0_reward)
    episode_rewards_single_p1.append(ep_p1_reward)

x = [1, 2, 3, 4, 5, 6]
y = [np.mean(episode_rewards_combined), np.mean(episode_rewards_combined_p0), np.mean(episode_rewards_combined_p1),
     np.mean(episode_rewards_single), np.mean(episode_rewards_single_p0), np.mean(episode_rewards_single_p1)]

y_std = [np.std(episode_rewards_combined), np.std(episode_rewards_combined_p0), np.std(episode_rewards_combined_p1),
         np.std(episode_rewards_single), np.std(episode_rewards_single_p0), np.std(episode_rewards_single_p1)]
width = 10
height = 8
plt.figure(figsize=(width, height))
plt.bar(x, y)
plt.title('Summed Reward of both Agents in the Particle Environment')
plt.xlabel('Left Combined Q function, Right only task Q function')
plt.ylabel('Average Team Reward over 100 episodes')
plt.errorbar(x, y, y_std, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
plt.savefig('figure.png', dpi=400, transparent=True)
plt.show()

