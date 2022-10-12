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

with open("impact_learner/all_ego_task.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open("impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

interaction_agent.switch_mode(ParticleInteractionAgent.LEARN_INTERACTION)
other_agent.switch_mode(ParticleInteractionAgent.INFERENCE)

ep_rews = {}
eval_scores = {}

for impact_task, task in zip(impact_tasks, tasks):

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[impact_task, task],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.add_interaction(impact_task)
    interaction_agent.switch_active_tasks([impact_task])

    other_agent.switch_active_tasks([task])

    episode_reward = 0
    num_eval_runs = 100

    action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
    action_dists = []

    episode_rewards = []
    average_q = []

    action_bag = []
    kwargs_agents = [{"self_agent_num": 0, "other_agent_nums": [1]}, {}]
    evaluation_scores = [evaluate(env, 10, [interaction_agent, other_agent], kwargs_agents=kwargs_agents)]

    for epoch in tqdm.tqdm(range(num_epochs)):
        state = env.reset()
        episode_reward = 0
        for time_steps in range(max_steps):
            action = interaction_agent.select_action(state["player_0"], self_agent_num=0, other_agent_nums=[1])
            action_2 = other_agent.select_action(state["player_1"])
            actions = {"player_0": action, "player_1": action_2}
            next_state, reward, done, _ = env.step(actions)
            episode_reward += reward["player_1"]

            interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                             reward["player_1"], done["player_0"])

            action_distribution[action] += 1

            if time_steps % 4 == 0:
                interaction_agent.learn_step(other_agent_num=1, self_agent_num=0,
                                             other_agent_model=other_agent.get_active_task_model())

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
            evaluation_scores.append(evaluate(env, 10, [interaction_agent, other_agent], kwargs_agents=kwargs_agents))
            # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
            # print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
    print(episode_rewards)
    ep_rews[task] = episode_rewards
    eval_scores[task] = evaluation_scores

stats = {"ep_rews": ep_rews, "eval_scores": eval_scores}
with open("stats/training_impact_learners2.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
interaction_agent.save_agent("impact_learner/all_ego_and_impact_task.agent")
