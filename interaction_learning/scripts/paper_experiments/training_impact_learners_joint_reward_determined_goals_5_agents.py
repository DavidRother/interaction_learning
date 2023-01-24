from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import make_deterministic
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
device = torch.device("cpu")

task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
task_2 = ["ta0", "ta1", "ta2", "ta3", "ta4"]
task_3 = ["tb2", "tb3", "tb4", "tb0", "tb1"]
task_4 = ["tc4", "tc0", "tc1", "tc2", "tc3"]
task_5 = ["te1", "td2", "te3", "td4", "te0"]

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up


agent_position_generator = lambda: [np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])]
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 0
render = False
num_agents = 5

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs_dim = env.observation_spaces["player_0"].shape[0]
obs_dim_impact = 8
n_actions = env.action_spaces["player_0"].n
task_alpha = 0.2
impact_alpha = 0.2
batch_size = 32
gamma = 0.50
target_update_interval = 1000
memory_size = 50000
action_alignment = False

num_epochs = 300

interaction_agent = ParticleInteractionAgent(obs_dim, obs_dim_impact, n_actions, task_alpha, impact_alpha,
                                             action_alignment, batch_size, gamma, target_update_interval,
                                             memory_size, device)

with open("impact_learner/all_ego_task_5_agents_no_ghosts.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open("impact_learner/all_ego_task_5_agents_no_ghosts.agent", "rb") as input_file:
    other_agent_2 = pickle.load(input_file)

with open("impact_learner/all_ego_task_5_agents_no_ghosts.agent", "rb") as input_file:
    other_agent_3 = pickle.load(input_file)

with open("impact_learner/all_ego_task_5_agents_no_ghosts.agent", "rb") as input_file:
    other_agent_4 = pickle.load(input_file)

interaction_agent.switch_mode(ParticleInteractionAgent.LEARN_TASK)
other_agent.switch_mode(ParticleInteractionAgent.INFERENCE)

ep_rews = {}
eval_scores = {}

for t1, t2, t3, t4, t5 in zip(task_1, task_2, task_3, task_4, task_5):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[t1, t2, t3, t4, t5],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.add_task(t1)
    interaction_agent.switch_active_tasks([t1])

    other_agent.switch_active_tasks([t2])
    other_agent_2.switch_active_tasks([t3])
    other_agent_3.switch_active_tasks([t4])
    other_agent_4.switch_active_tasks([t5])

    episode_reward = 0
    num_eval_runs = 100

    action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
    action_dists = []

    episode_rewards = []
    average_q = []

    action_bag = []
    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1, 2, 3, 4]}, {}, {}, {}, {}]
    evaluation_scores = [evaluate(env, 10, [interaction_agent, other_agent, other_agent, other_agent, other_agent], kwargs_agents=kwargs_agents)]

    for epoch in tqdm.tqdm(range(num_epochs)):
        state = env.reset()
        episode_reward = 0
        for time_steps in range(max_steps):
            action = interaction_agent.select_action(state["player_0"], self_agent_num=0, other_agent_nums=[1, 2, 3, 4])
            action_2 = other_agent.select_action(state["player_1"])
            action_3 = other_agent_2.select_action(state["player_2"])
            action_4 = other_agent_3.select_action(state["player_3"])
            action_5 = other_agent_4.select_action(state["player_4"])
            actions = {"player_0": action, "player_1": action_2, "player_2": action_3, "player_3": action_4,
                       "player_4": action_5}
            next_state, reward, done, _ = env.step(actions)
            episode_reward += reward["player_1"]

            interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                             (reward["player_1"] + reward["player_0"] + reward["player_2"] + reward["player_3"] + reward["player_4"]) / 25, done["player_0"])

            action_distribution[action] += 1

            if time_steps % 4 == 0:
                interaction_agent.learn_step()

            if all(done.values()):
                break

            state = next_state
            if render:
                env.render(mode="human")
                sleep(0.1)
        if episode_reward > 0:
            print(f"Reward higher than anticipated: {action_bag}")
        action_bag = []
        episode_rewards.append(episode_reward)
        if epoch % 10 == 0:
            evaluation_scores.append(evaluate(env, 10, [interaction_agent, other_agent, other_agent, other_agent, other_agent], kwargs_agents=kwargs_agents))
            # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
            # print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
    print(episode_rewards)
    ep_rews[t1] = episode_rewards
    eval_scores[t2] = evaluation_scores

stats = {"ep_rews": ep_rews, "eval_scores": eval_scores}
with open("stats/training_impact_learners_joint_reward_determined_goals_5_agents.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
interaction_agent.save_agent("impact_learner/joint_learner_determined_goals_5_agents.agent")
