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


device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

agent_position_generator = lambda: [np.asarray([0.05, np.random.uniform(0, 1)])]
agent_reward = ["y0"]
max_steps = 1000
ghost_agents = 1
render = True

alpha = 0.04

env = GymPartiEnvironment(agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                          max_steps=max_steps, ghost_agents=ghost_agents, render=render)
# temperature = torch.Tensor(alpha)
onlineQNetwork = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, alpha, device="cpu").to(device)
targetQNetwork = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, alpha, device="cpu").to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)
# entropy_optimizer = torch.optim.Adam(temperature, lr=1e-4)

GAMMA = 0.80
REPLAY_MEMORY = 50000
BATCH = 32
UPDATE_STEPS = 1000

memory_replay = Memory(REPLAY_MEMORY)

learn_steps = 0
begin_learn = False
episode_reward = 0

action_distribution = {n: 0 for n in range(env.action_space.n)}
action_dists = []

episode_rewards = []
average_q = []

action_bag = []

for epoch in range(500):
    state = env.reset()
    episode_reward = 0
    for time_steps in range(max_steps):
        action = onlineQNetwork.select_action(state)
        action_bag.append(action)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        memory_replay.add((state, next_state, action, reward, done))

        action_distribution[action] += 1

        if memory_replay.size() > 1000 and time_steps % 4 == 0:
            if begin_learn is False:
                print('learning begins!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                next_q = targetQNetwork(batch_next_state)
                next_v = targetQNetwork.get_value(next_q)
                y = batch_reward + (1 - batch_done) * GAMMA * next_v

                average_q.append(torch.mean(next_q).cpu().item())

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            if torch.isinf(loss).any().cpu().item():
                print("hi")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # temperature_loss = 0

        if done:
            dist = np.asarray(list(action_distribution.values())) / sum(action_distribution.values())
            action_dists.append(dist)
            print(f"Action  Dsitribution: {dist}")
            action_distribution = {n: 0 for n in range(env.action_space.n)}
            break

        state = next_state
    if episode_reward > -43.1:
        print(f"Reward higher than anticipated: {action_bag}")
    action_bag = []
    print(episode_reward)
    episode_rewards.append(episode_reward)
    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), f'sql{epoch}policy_y0')
        print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

plt.figure(1)
plt.plot(episode_rewards)

plt.figure(2)
plt.plot(action_dists)
plt.legend(["NO OP", "RIGHT", "DOWN", "LEFT", "UP"])

plt.figure(3)
plt.plot(average_q)
plt.title("Average encountered Q value in training")

plt.show()