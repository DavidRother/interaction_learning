from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.partigame import parallel_env
from interaction_learning.core.util import make_deterministic, AgentPositionGenerator
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch
import tqdm
import pickle


make_deterministic(1)


device = torch.device("cpu")

task_1 = ["ta2"]
task_2 = ["ta0"]
task_3 = ["tb2"]
task_4 = ["tc4"]
task_5 = ["te1"]

impact_tasks_1 = [[t1.replace("t", "i"), t2.replace("t", "i"),
                   t3.replace("t", "i"), t4.replace("t", "i")]
                  for t1, t2, t3, t4 in zip(task_2, task_3, task_4, task_5)]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

eval_scores = {alg: {} for alg in algorithms}

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

num_eval = 100

agent_position_generator = AgentPositionGenerator(num_eval * 10, num_agents=5)
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 0
render = False
num_agents = 5

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents.agent", "rb") as input_file:
    ego_agent = pickle.load(input_file)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents_2_stage.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open("../paper_experiments/impact_learner/joint_learner_determined_goals_5_agents.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents.agent", "rb") as input_file:
    other_agent_1 = pickle.load(input_file)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents.agent", "rb") as input_file:
    other_agent_2 = pickle.load(input_file)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents.agent", "rb") as input_file:
    other_agent_3 = pickle.load(input_file)

with open("../paper_experiments/impact_learner/selected_tasks_5_agents.agent", "rb") as input_file:
    other_agent_4 = pickle.load(input_file)


interaction_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
ego_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
joint_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent_1.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent_2.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent_3.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent_4.switch_mode(ParticleInteractionAgent.INFERENCE)

########################################################################################################################
# Test aligned interaction learner #####################################################################################
########################################################################################################################

interaction_agent.action_alignment = True

algorithm = "action_aligned_interaction_learner"

agent_position_generator.reset()

for t1, t2, t3, t4, t5, i1 in zip(task_1, task_2, task_3, task_4, task_5, impact_tasks_1):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2, t3, t4, t5],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([*i1, t1])
    other_agent_1.switch_active_tasks([t2])
    other_agent_2.switch_active_tasks([t3])
    other_agent_3.switch_active_tasks([t4])
    other_agent_4.switch_active_tasks([t5])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1, 2, 3, 4]}, {}, {}, {}, {}]
    evaluation_scores = [evaluate(env, num_eval, [interaction_agent, other_agent_1, other_agent_2,
                                                  other_agent_3, other_agent_4], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + ''.join(i1) + t2 + t3 + t4 + t5] = evaluation_scores

########################################################################################################################
# Test Non aligned interaction learner #################################################################################
########################################################################################################################

interaction_agent.action_alignment = False

algorithm = "non_aligned_interaction_learner"

agent_position_generator.reset()

for t1, t2, t3, t4, t5, i1 in zip(task_1, task_2, task_3, task_4, task_5, impact_tasks_1):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2, t3, t4, t5],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([*i1, t1])
    other_agent_1.switch_active_tasks([t2])
    other_agent_2.switch_active_tasks([t3])
    other_agent_3.switch_active_tasks([t4])
    other_agent_4.switch_active_tasks([t5])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1, 2, 3, 4]}, {}, {}, {}, {}]
    evaluation_scores = [evaluate(env, num_eval, [interaction_agent, other_agent_1, other_agent_2,
                                                  other_agent_3, other_agent_4], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + ''.join(i1) + t2 + t3 + t4 + t5] = evaluation_scores

########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "joint_learner"

agent_position_generator.reset()

for t1, t2, t3, t4, t5, i1 in zip(task_1, task_2, task_3, task_4, task_5, impact_tasks_1):

    joint_agent.switch_active_tasks([t1])
    other_agent_1.switch_active_tasks([t2])
    other_agent_2.switch_active_tasks([t3])
    other_agent_3.switch_active_tasks([t4])
    other_agent_4.switch_active_tasks([t5])

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2, t3, t4, t5],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1, 2, 3, 4]}, {}, {}, {}, {}]
    evaluation_scores = [evaluate(env, num_eval, [joint_agent, other_agent_1, other_agent_2,
                                                  other_agent_3, other_agent_4], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + ''.join(i1) + t2 + t3 + t4 + t5] = evaluation_scores

########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "selfish_task_solver"

agent_position_generator.reset()

for t1, t2, t3, t4, t5, i1 in zip(task_1, task_2, task_3, task_4, task_5, impact_tasks_1):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2, t3, t4, t5],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    ego_agent.switch_active_tasks([t2])
    other_agent_1.switch_active_tasks([t2])
    other_agent_2.switch_active_tasks([t3])
    other_agent_3.switch_active_tasks([t4])
    other_agent_4.switch_active_tasks([t5])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1, 2, 3, 4]}, {}, {}, {}, {}]
    evaluation_scores = [evaluate(env, num_eval, [ego_agent, other_agent_1, other_agent_2,
                                                  other_agent_3, other_agent_4], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + ''.join(i1) + t2 + t3 + t4 + t5] = evaluation_scores

########################################################################################################################
# Save Results #########################################################################################################
########################################################################################################################

stats = {"eval_scores": eval_scores}
with open("stats/test_selected_non_aligned_scenarios_many_agents_no_collision.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
