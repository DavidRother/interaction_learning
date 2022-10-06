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
agent_reward = ["y0d", ""]
max_steps = 1000
ghost_agents = 0
render = False
num_agents = 2

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)
# temperature = torch.Tensor(alpha)
alpha = 0.1
onlineQNetwork = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                              alpha, device="cpu").to(device)

onlineQNetwork.load_state_dict(torch.load("agent/sql_final_policy_y0d"))

agent2net = SoftQNetwork(env.observation_spaces["player_1"].shape[0], env.action_spaces["player_1"].n,
                         alpha, device="cpu").to(device)

impact_alpha = 0.3
impact_net = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                          impact_alpha, device="cpu").to(device)
target_impact_net = SoftQNetwork(env.observation_spaces["player_0"].shape[0], env.action_spaces["player_0"].n,
                                 impact_alpha, device="cpu").to(device)

target_impact_net.load_state_dict(impact_net.state_dict())

impact_optimizer = torch.optim.Adam(impact_net.parameters(), lr=1e-4)

agent2net.load_state_dict(torch.load("agent/sql490policy"))
# entropy_optimizer = torch.optim.Adam(temperature, lr=1e-4)

GAMMA = 0.80
REPLAY_MEMORY = 50000
BATCH = 32
UPDATE_STEPS = 1000

memory_replay = Memory(REPLAY_MEMORY)

learn_steps = 0
begin_learn = False
episode_reward = 0

action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
action_dists = []

episode_rewards = []
average_q = []

action_bag = []

for epoch in range(400):
    state = env.reset()
    episode_reward = 0
    for time_steps in range(max_steps):
        action = onlineQNetwork.select_action(state["player_0"])
        action_bag.append(action)
        action_2 = impact_net.select_action(state["player_1"])
        actions = {"player_0": action, "player_1": action_2}
        next_state, reward, done, _ = env.step(actions)
        episode_reward += reward["player_0"]

        memory_replay.add((state["player_0"], next_state["player_0"], action_2, reward["player_0"], done["player_0"]))

        action_distribution[action] += 1

        if memory_replay.size() > 1000 and time_steps % 4 == 0:
            if begin_learn is False:
                print('learning begins!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                target_impact_net.load_state_dict(impact_net.state_dict())

            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(np.asarray(batch_state)).to(device)
            batch_next_state = torch.FloatTensor(np.asarray(batch_next_state)).to(device)
            batch_action = torch.FloatTensor(np.asarray(batch_action)).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(np.asarray(batch_reward)).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(np.asarray(batch_done)).unsqueeze(1).to(device)

            with torch.no_grad():
                current_q = onlineQNetwork(batch_state)
                current_v = onlineQNetwork.get_value(current_q)
                next_q = onlineQNetwork(batch_next_state)
                next_v = onlineQNetwork.get_value(next_q)
                future_impact_q = target_impact_net(batch_next_state)
                future_impact_value = target_impact_net.get_value(future_impact_q)
                r = next_v + batch_reward - current_v
                y = r + (1 - batch_done) * GAMMA * future_impact_value
                delta = current_v - next_v

                average_q.append(torch.mean(future_impact_q).cpu().item())

            loss = F.mse_loss(impact_net(batch_state).gather(1, batch_action.long()), y)
            if torch.isinf(loss).any().cpu().item():
                print("hi")
            impact_optimizer.zero_grad()
            loss.backward()
            impact_optimizer.step()

            # temperature_loss = 0

        if done["player_0"]:
            dist = np.asarray(list(action_distribution.values())) / sum(action_distribution.values())
            action_dists.append(dist)
            print(f"Action  Distribution: {dist}")
            action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
            break

        state = next_state
        if render:
            env.render(mode="human")
            sleep(0.1)
    if episode_reward >= 0:
        print(f"Reward higher than anticipated: {action_bag}")
    action_bag = []
    print(episode_reward)
    episode_rewards.append(episode_reward)
    if epoch % 10 == 0:
        # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
        print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
torch.save(impact_net.state_dict(), f'agent/sql_final_policy_y0d_impact')

plt.figure(1)
plt.plot(episode_rewards)

plt.figure(2)
plt.plot(action_dists)
plt.legend(["NO OP", "RIGHT", "DOWN", "LEFT", "UP"])

plt.figure(3)
plt.plot(average_q)
plt.title("Average encountered Q value in training")

plt.show()
