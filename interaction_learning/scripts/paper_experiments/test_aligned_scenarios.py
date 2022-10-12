from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch
import tqdm
import pickle

x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
impact_tasks = ["i" + x + y for x in x_tasks for y in y_tasks] + ["i" + x for x in x_tasks] + ["i" + y for y in y_tasks]
impact_tasks = [impact_tasks[0]]
device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up


agent_position_generator = lambda: [np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])]
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 0
render = False
num_agents = 2

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs_dim = env.observation_spaces["player_0"].shape[0]
n_actions = env.action_spaces["player_0"].n
task_alpha = 0.05
impact_alpha = 0.05
batch_size = 32
gamma = 0.5
target_update_interval = 1000
memory_size = 50000

num_epochs = 200

with open("impact_learner/all_ego_and_impact_task.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open("impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

interaction_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent.switch_mode(ParticleInteractionAgent.INFERENCE)

interaction_agent.action_alignment = True

eval_scores = {}

for impact_task, task in zip(impact_tasks, tasks):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=["te", task],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([impact_task, "te"])

    other_agent.switch_active_tasks([task])
    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[task] = evaluation_scores

stats = {"eval_scores": eval_scores}
with open("stats/test_aligned_scenarios.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
