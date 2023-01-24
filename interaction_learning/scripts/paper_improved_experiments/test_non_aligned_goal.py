from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import make_deterministic, AgentPositionGenerator
import torch
import pickle


make_deterministic(1)

device = torch.device("cpu")

aligned_task_1 = ["ta2"]
impact_task_1 = ["ib3"]
aligned_task_2 = ["tb3"]

# aligned_task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
# impact_task_1 = ["ib3", "ic4", "ia0", "id2", "ie1"]
# aligned_task_2 = ["tb3", "tc4", "ta0", "td2", "te1"]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

eval_scores = {alg: {} for alg in algorithms}

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

num_eval = 200
stats_location = "stats/test_misaligned_scenarios.stats"
# , x_min=0.4, x_max=0.6, y_min=0.4, y_max=0.6
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

with open("../paper_experiments/impact_learner/all_ego_and_impact_task_longer_train.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open(f"../paper_experiments/impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"../paper_experiments/impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
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

print(algorithm)
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

print(algorithm)
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

print(algorithm)
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

print(algorithm)
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
with open(stats_location, 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
