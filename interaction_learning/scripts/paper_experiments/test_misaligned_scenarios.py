from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import make_deterministic, AgentPositionGenerator
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch
import tqdm
import pickle


make_deterministic(1)

x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
impact_tasks = ["i" + x + y for x in x_tasks for y in y_tasks] + ["i" + x for x in x_tasks] + ["i" + y for y in y_tasks]
impact_tasks = [impact_tasks[0]]
device = torch.device("cpu")

aligned_task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
impact_task_1 = ["ib3", "ic4", "ia0", "id2", "ie1"]
aligned_task_2 = ["tb3", "tc4", "ta0", "td2", "te1"]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

eval_scores = {alg: {} for alg in algorithms}

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

num_eval = 100

agent_position_generator = AgentPositionGenerator(num_eval * 10)
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

with open(f"impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)

interaction_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
joint_agent.switch_mode(ParticleInteractionAgent.INFERENCE)

########################################################################################################################
# Test aligned interaction learner #####################################################################################
########################################################################################################################

interaction_agent.action_alignment = True

algorithm = "action_aligned_interaction_learner"

agent_position_generator.reset()

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([i1, t1])
    other_agent.switch_active_tasks([t2])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, num_eval, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Test Non aligned interaction learner #################################################################################
########################################################################################################################

interaction_agent.action_alignment = False

algorithm = "non_aligned_interaction_learner"

agent_position_generator.reset()

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([i1, t1])
    other_agent.switch_active_tasks([t2])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, num_eval, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores


########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "selfish_task_solver"

agent_position_generator.reset()

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([t1])
    other_agent.switch_active_tasks([t2])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, num_eval, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores


########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "joint_learner"

agent_position_generator.reset()

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    joint_agent.switch_active_tasks([t1])
    other_agent.switch_active_tasks([t2])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, num_eval, [joint_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Save Results #########################################################################################################
########################################################################################################################

stats = {"eval_scores": eval_scores}
with open("stats/test_misaligned_scenarios.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)