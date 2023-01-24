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

with open("impact_learner/all_ego_and_impact_task.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open(f"impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)

task = "ta2"

# Koordinaten und Richtungen definieren
x, y = np.meshgrid(np.arange(0.05, 0.95, .05), np.arange(0.05, 0.95, .05))
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
        q_task = joint_agent.task_models[task].get_q(state)
        dist_task = joint_agent.task_models[task].get_dist(q_task)
        u[a, b] = dist_task.data[0, 1].item() - dist_task.data[0, 3].item()
        v[a, b] = dist_task.data[0, 2].item() - dist_task.data[0, 4].item()


# Diagramme definieren
fig, ax = plt.subplots()
# Quiver-Diagramm erstellen
ax.quiver(x, y, u, v)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.margins(y=0)
ax.margins(x=0)
# ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis('equal')
fig.tight_layout()
ax.set_title("JL")


task = "ta2"

# Koordinaten und Richtungen definieren
x, y = np.meshgrid(np.arange(0.05, 0.95, .05), np.arange(0.05, 0.95, .05))
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
        q_task = interaction_agent.task_models[task].get_q(state)
        dist_task = interaction_agent.task_models[task].get_dist(q_task)
        u[a, b] = dist_task.data[0, 1].item() - dist_task.data[0, 3].item()
        v[a, b] = dist_task.data[0, 2].item() - dist_task.data[0, 4].item()


fig2, ax2 = plt.subplots()
# Quiver-Diagramm erstellen
ax2.quiver(x, y, u, v)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.margins(y=0)
ax2.margins(x=0)
# ax.set(xlim=(0, 1), ylim=(0, 1))
ax2.axis('equal')
fig2.tight_layout()
ax2.set_title("SL")

task = "ta2"
impact_task = "ib3"

# Koordinaten und Richtungen definieren
x, y = np.meshgrid(np.arange(0.05, 0.95, .05), np.arange(0.05, 0.95, .05))
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
        q_task = interaction_agent.task_models[task].get_q(state)
        dist_task = interaction_agent.task_models[task].get_dist(q_task)

        q_interactions = [interaction_agent.interaction_models[impact_task].get_q(state, 0, 1)]
        weights = torch.FloatTensor([1./(1 + len(q_interactions))] * (1 + len(q_interactions)))

        q_values = [q_task] + [q for q in q_interactions]
        comb_q = []
        for q, w in zip(q_values, weights):
            comb_q.append(q * w.item())
        combined_q = torch.Tensor(sum(comb_q))
        combined_v = interaction_agent.task_models[task].get_v(combined_q)

        dist = torch.exp((combined_q - combined_v) / interaction_agent.task_alpha)

        u[a, b] = dist.data[0, 1].item() - dist.data[0, 3].item()
        v[a, b] = dist.data[0, 2].item() - dist.data[0, 4].item()


fig3, ax3 = plt.subplots()
# Quiver-Diagramm erstellen
ax3.quiver(x, y, u, v)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.margins(y=0)
ax3.margins(x=0)
# ax.set(xlim=(0, 1), ylim=(0, 1))
ax3.axis('equal')
fig3.tight_layout()
ax3.set_title("NAIL")


task = "ta2"
impact_task = "ib3"

# Koordinaten und Richtungen definieren
x, y = np.meshgrid(np.arange(0.05, 0.95, .05), np.arange(0.05, 0.95, .05))
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        with torch.no_grad():
            state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
            q_task = interaction_agent.task_models[task].get_q(state)
            dist_task = interaction_agent.task_models[task].get_dist(q_task)

            q_interactions = [interaction_agent.interaction_models[impact_task].get_q(state, 0, 1)]

            dist_interactions = [interaction_agent.interaction_models[impact_task].get_dist(q_interactions[0])]
            dists = [dist_task.cpu().numpy().flatten()] + [d.cpu().numpy().flatten() for d in dist_interactions]
            entropies = [1 - interaction_agent.calc_entropy(inter) for inter in dists]
            r_ent = [1 - interaction_agent.calc_relative_entropy(dist, dists[0]) for dist in dists]
            support = len(dists[0])
            linear_entropies = [np.power(support, ent) / support for ent in entropies]
            linear_r_ent = [np.power(support, ent) / support for ent in r_ent]
            ent_weights = torch.Tensor([l_ent * l_r_ent for l_ent, l_r_ent in zip(linear_entropies, linear_r_ent)])
            weights = ent_weights / sum(ent_weights)

            q_values = [q_task] + [q for q in q_interactions]
            comb_q = []
            for q, w in zip(q_values, weights):
                comb_q.append(q * w.item())
            combined_q = torch.Tensor(sum(comb_q))
            combined_v = interaction_agent.task_models[task].get_v(combined_q)

            dist = torch.exp((combined_q - combined_v) / interaction_agent.task_alpha)

            u[a, b] = dist.data[0, 1].item() - dist.data[0, 3].item()
            v[a, b] = dist.data[0, 2].item() - dist.data[0, 4].item()


fig4, ax4 = plt.subplots()
# Quiver-Diagramm erstellen
ax4.quiver(x, y, u, v)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.margins(y=0)
ax4.margins(x=0)
# ax.set(xlim=(0, 1), ylim=(0, 1))
ax4.axis('equal')
fig4.tight_layout()

ax4.set_title("AAIL")

# Quiver-Diagramm anzeigen
plt.show()
