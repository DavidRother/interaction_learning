from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import make_deterministic
import numpy as np
import torch
import pickle


make_deterministic(1)

x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
impact_tasks = ["i" + x + y for x in x_tasks for y in y_tasks] + ["i" + x for x in x_tasks] + ["i" + y for y in y_tasks]
task_impact_map = {t: i for t, i in zip(tasks, impact_tasks)}
device = torch.device("cpu")

aligned_task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
impact_task_1 = ["ib3", "ic4", "ia0", "id2", "ie1"]
aligned_task_2 = ["tb3", "tc4", "ta0", "td2", "te1"]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

eval_scores = {alg: {} for alg in algorithms}

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

interaction_agent.switch_mode(ParticleInteractionAgent.INFERENCE)

########################################################################################################################
# Test aligned interaction learner #####################################################################################
########################################################################################################################

interaction_agent.action_alignment = True

algorithm = "action_aligned_interaction_learner"

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):
    with open(f"ppo_other_agents/ppo_other_agent_{t2}", "rb") as input_file:
        other_agent = pickle.load(input_file)

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([i1, t1])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Test Non aligned interaction learner #################################################################################
########################################################################################################################

interaction_agent.action_alignment = False

algorithm = "non_aligned_interaction_learner"

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):
    with open(f"ppo_other_agents/ppo_other_agent_{t2}", "rb") as input_file:
        other_agent = pickle.load(input_file)

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.switch_active_tasks([i1, t1])

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "ppo_joint_learner"

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):
    with open(f"ppo_joint_learner/ppo_joint_agent_{t1 + i1}", "rb") as input_file:
        ppo_agent = pickle.load(input_file)
    with open(f"ppo_other_agents/ppo_other_agent_{t2}", "rb") as input_file:
        other_agent = pickle.load(input_file)

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [ppo_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Test PPO #############################################################################################################
########################################################################################################################

algorithm = "ppo_single_learner"

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):
    with open(f"ppo_learner/ppo_single_learner_{t1}", "rb") as input_file:
        ppo_agent = pickle.load(input_file)
    with open(f"ppo_other_agents/ppo_other_agent_{t2}", "rb") as input_file:
        other_agent = pickle.load(input_file)

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [ppo_agent, other_agent], kwargs_agents=kwargs_agents)]
    eval_scores[algorithm][t1 + i1 + t2] = evaluation_scores

########################################################################################################################
# Save Results #########################################################################################################
########################################################################################################################

stats = {"eval_scores": eval_scores}
with open("stats/test_aligned_scenarios.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
