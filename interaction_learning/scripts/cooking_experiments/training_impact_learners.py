from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate, DummyAgent
from gym_cooking.environment.cooking_env import parallel_env
from interaction_learning.core.util import make_deterministic
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch
import tqdm
import pickle


make_deterministic(1)

device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up


n_agents = 1
num_humans = 0
max_steps = 100
render = True
obs_spaces = None
action_scheme = "scheme1"
ghost_agents = 0
manual_control = False

level = 'open_room_salad2'
seed = 1
record = False
max_num_timesteps = 1000
recipe = "TomatoSalad"
recipes = [recipe]

env = parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps, recipes=recipes,
                   obs_spaces=obs_spaces, action_scheme=action_scheme, ghost_agents=ghost_agents, render=render)

obs_dim = env.observation_spaces["player_0"].shape[0]
obs_dim_impact = 8
n_actions = env.action_spaces["player_0"].n
task_alpha = 0.04
impact_alpha = 0.04
batch_size = 32
gamma = 0.50
target_update_interval = 1000
memory_size = 50000
action_alignment = False

num_epochs = 400

interaction_agent = ParticleInteractionAgent(obs_dim, obs_dim_impact, n_actions, task_alpha, impact_alpha,
                                             action_alignment, batch_size, gamma, target_update_interval,
                                             memory_size, device)

dummy_agent = DummyAgent()
ep_rews = {}
eval_scores = {}
ep_length_stats = {}

env = parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps, recipes=recipes,
                   obs_spaces=obs_spaces, action_scheme=action_scheme, ghost_agents=ghost_agents, render=render)

interaction_agent.add_task(recipe)
interaction_agent.switch_active_tasks([recipe])

episode_reward = 0
num_eval_runs = 10

action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
action_dists = []

episode_rewards = []
episode_lengths = []
average_q = []

action_bag = []

evaluation_scores = [evaluate(env, 10, [interaction_agent])]

for epoch in tqdm.tqdm(range(num_epochs)):
    state = env.reset()
    episode_reward = 0
    ep_length = 0
    for time_steps in range(max_steps):
        action = interaction_agent.select_action(state["player_0"])
        actions = {"player_0": action, "player_1": 0}
        next_state, reward, done, truncation, info = env.step(actions)
        episode_reward += reward["player_0"]

        interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                         reward["player_0"], done["player_0"])

        action_distribution[action] += 1
        ep_length += 1

        if time_steps % 4 == 0:
            interaction_agent.learn_step()

        if all(done.values()):
            break

        state = next_state

        if render:
            env.render()
            sleep(0.1)
    # if episode_reward > 1000:
    #     print(f"Reward higher than anticipated: {episode_reward}")
    action_bag = []
    episode_rewards.append(episode_reward)
    episode_lengths.append(ep_length)
    print(episode_rewards)
    # print(f"Episode Rewards: {episode_reward} || Episode Length: {ep_length}")
    if epoch % 10 == 0:
        evaluation_scores.append(evaluate(env, 10, [interaction_agent, dummy_agent]))
        # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
        # print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

