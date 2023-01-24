from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import  AgentPositionGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

plt.style.use("seaborn-whitegrid")

num_eval = 100

agent_position_generator = AgentPositionGenerator(num_eval * 10)
agent_reward = ["x"]
max_steps = 1000
ghost_agents = 0
render = False
num_agents = 2

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()
p_obs = obs["player_0"]
p_obs[3] = 0.1
p_obs[4] = 0.9

with open("impact_learner/all_ego_and_impact_task.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open(f"impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)

task = "ta"
impact_task = "ie0"

x, y = np.meshgrid(np.arange(0.0, 1.025, .025), np.arange(0.0, 1.025, .05))

# Koordinaten und Richtungen definieren
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        # print(f"x: {x[a][b]} || y: {y[a][b]}")
        with torch.no_grad():
            state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
            q_task = interaction_agent.task_models[task].get_q(state)
            dist_task = interaction_agent.task_models[task].get_dist(q_task).cpu().numpy().flatten()
            support = len(dist_task)
            entropy = interaction_agent.calc_entropy(dist_task)
            linear_entropy = 1 - np.power(support, entropy) / support
            u[a][b] = linear_entropy


fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, u, cmap='hot')
ax.axis([0, 1, 0, 1])
ax.set_title("Task Entropy")
fig.colorbar(c, ax=ax)
plt.show()
