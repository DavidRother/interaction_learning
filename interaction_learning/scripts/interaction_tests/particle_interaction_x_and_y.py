from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate, DummyAgent
from partigames.environment.zoo_env import parallel_env
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch

tasks = ["ty0d", "ty1d", "ty2d", "ty3d", "ty4d", "tx"]
interaction_tasks = ["iy0d", "iy1d", "iy2d", "iy3d", "iy4d"]
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

interaction_agent = ParticleInteractionAgent(obs_dim, n_actions, task_alpha, impact_alpha, batch_size, gamma,
                                             target_update_interval, memory_size, device)

dummy_agent = DummyAgent()

plt.style.use("ggplot")
fig_idx = 1

for task in tasks:

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[task, ""], max_steps=max_steps, ghost_agents=ghost_agents, render=render)

    interaction_agent.add_task(task)
    interaction_agent.switch_active_tasks([task])

    episode_reward = 0
    num_eval_runs = 100

    evaluation_scores = []

    for epoch in range(400):
        state = env.reset()
        episode_reward = 0
        for time_steps in range(max_steps):
            action = interaction_agent.select_action(state["player_0"])
            actions = {"player_0": action, "player_1": 0}
            next_state, reward, done, _ = env.step(actions)
            episode_reward += reward["player_0"]

            interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                             reward["player_0"], done["player_0"])

            if time_steps % 4 == 0:
                interaction_agent.learn_step()

            if done["player_0"]:
                break

            state = next_state
        if epoch % 10 == 0:
            evaluation_scores.append(evaluate(env, 10, [interaction_agent, dummy_agent]))
            # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
            print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

    interaction_agent.save_agent("interaction_agents/all_tasks.agent")

    eval_mean = []
    eval_std = []

    for eval_scores in evaluation_scores:

        eval_mean.append(np.mean([evaluations["player_0"] for evaluations in eval_scores]))
        eval_std.append(np.std([evaluations["player_0"] for evaluations in eval_scores]))

    eval_mean = np.asarray(eval_mean)
    eval_std = np.asarray(eval_std)

    plt.figure(fig_idx)
    plt.plot(eval_mean)
    plt.fill_between(list(range(len(eval_std))), eval_mean + eval_std, eval_mean - eval_std, alpha=0.5)
    plt.title(f"Evaluation Scores during training for task {task}")
    plt.xlabel("Evaluation Epoch (each 10th Training Episode)")
    plt.ylabel("Reward")
    fig_idx += 1

plt.show()
