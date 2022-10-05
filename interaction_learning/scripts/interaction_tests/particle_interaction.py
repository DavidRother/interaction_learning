from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from interaction_learning.core.evaluation import evaluate
from partigames.environment.zoo_env import parallel_env
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch

tasks = ["tx", "ty0d", "ty1d", "ty2d", "ty3d", "ty4d", "iy0d", "iy1d", "iy2d", "iy3d", "iy4d"]
device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

agent_position_generator = lambda: [np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])]
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 1
render = False
num_agents = 1

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs_dim = env.observation_spaces["player_0"].shape[0]
n_actions = env.action_spaces["player_0"].n
task_alpha = 0.05
impact_alpha = 0.05
batch_size = 32
gamma = 0.7
target_update_interval = 1000
memory_size = 50000

interaction_agent = ParticleInteractionAgent(obs_dim, n_actions, task_alpha, impact_alpha, batch_size, gamma,
                                             target_update_interval, memory_size, device)

interaction_agent.add_task("tx")
interaction_agent.switch_active_tasks(["tx"])

episode_reward = 0
num_eval_runs = 100

action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
action_dists = []

episode_rewards = []
average_q = []

action_bag = []

evaluation_scores = [evaluate(env, 10, [interaction_agent])]

for epoch in range(400):
    state = env.reset()
    episode_reward = 0
    for time_steps in range(max_steps):
        action = interaction_agent.select_action(state["player_0"])
        actions = {"player_0": action}
        next_state, reward, done, _ = env.step(actions)
        episode_reward += reward["player_0"]

        interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                         reward["player_0"], done["player_0"])

        action_distribution[action] += 1

        if time_steps % 4 == 0:
            interaction_agent.learn_step()

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
        evaluation_scores.append(evaluate(env, 10, [interaction_agent]))
        # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
        print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

interaction_agent.save_agent("interaction_agents/task_x.agent")

plt.style.use("ggplot")

plt.figure(1)
plt.plot(episode_rewards)

plt.figure(2)
plt.plot(action_dists)
plt.legend(["NO OP", "RIGHT", "DOWN", "LEFT", "UP"])

eval_mean = []
eval_std = []

for eval_scores in evaluation_scores:

    eval_mean.append(np.mean([evaluations["player_0"] for evaluations in eval_scores]))
    eval_std.append(np.std([evaluations["player_0"] for evaluations in eval_scores]))

eval_mean = np.asarray(eval_mean)
eval_std = np.asarray(eval_std)

plt.figure(3)
plt.plot(eval_mean)
plt.fill_between(list(range(len(eval_std))), eval_mean + eval_std, eval_mean - eval_std, alpha=0.5)
plt.title("Evaluation Scores during training")
plt.xlabel("Evaluation Epoch (each 10th Training Episode)")
plt.ylabel("Reward")


plt.show()
