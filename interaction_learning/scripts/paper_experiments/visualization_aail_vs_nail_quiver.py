from partigames.environment.zoo_env import parallel_env
from interaction_learning.core.util import make_deterministic, AgentPositionGenerator
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

make_deterministic(1)


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
# p_obs[4] = 0.0
# p_obs[5] = 0.0
# p_obs[6] = 0.0
# p_obs[7] = 0.0
print(p_obs)

with open("impact_learner/all_ego_and_impact_task_longer_train.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open(f"impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)
    

task = "ta1"
impact_task = "ib2"

fontsize = 24

# Koordinaten und Richtungen definieren
start = 0.0
end = 1.0
step = 0.1
x, y = np.meshgrid(np.arange(start, end + step, step), np.arange(start, end + step, step))
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")
        q_task = other_agent.task_models[task].get_q(state)
        dist_task = other_agent.task_models[task].get_dist(q_task)

        u[a, b] = dist_task.data[0, 1].item() - dist_task.data[0, 3].item()
        v[a, b] = dist_task.data[0, 2].item() - dist_task.data[0, 4].item()

fig, axs = plt.subplots(2, 2)
# Quiver-Diagramm erstellen
axs[0, 0].quiver(x, y, u, v, angles='xy')
axs[0, 0].axis([0, 1, 0, 1])
axs[0, 0].invert_yaxis()
# ax.set(xlim=(0, 1), ylim=(0, 1))
axs[0, 0].axis('equal')
axs[0, 0].set_title("Ego Task", fontsize=fontsize)

# Koordinaten und Richtungen definieren
z = np.zeros_like(x)
u = np.zeros_like(x)
v = np.zeros_like(x)
# 0 Still 1 right 2 down 3 left 4 up
for a in range(x.shape[0]):
    for b in range(x.shape[1]):
        p_obs[0] = x[a][b]
        p_obs[1] = y[a][b]
        state = torch.FloatTensor(p_obs).unsqueeze(0).to("cpu")

        q_interaction = interaction_agent.interaction_models[impact_task].get_q(state, 0, 1)
        combined_v = interaction_agent.interaction_models[impact_task].get_v(q_interaction)

        dist = torch.exp((q_interaction - combined_v) / interaction_agent.impact_alpha)
        dist /= torch.sum(dist)

        u[a, b] = dist.data[0, 1].item() - dist.data[0, 3].item()
        v[a, b] = dist.data[0, 2].item() - dist.data[0, 4].item()


# Quiver-Diagramm erstellen
axs[0, 1].quiver(x, y, u, v, angles='xy')
axs[0, 1].axis([0, 1, 0, 1])
axs[0, 1].invert_yaxis()
# ax.set(xlim=(0, 1), ylim=(0, 1))
axs[0, 1].axis('equal')
axs[0, 1].set_title("Impact Task", fontsize=fontsize)

# Koordinaten und Richtungen definieren
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
        dist /= torch.sum(dist)

        u[a, b] = dist.data[0, 1].item() - dist.data[0, 3].item()
        v[a, b] = dist.data[0, 2].item() - dist.data[0, 4].item()


# Quiver-Diagramm erstellen
axs[1, 0].quiver(x, y, u, v, angles='xy')
axs[1, 0].axis([0, 1, 0, 1])
axs[1, 0].invert_yaxis()
# ax.set(xlim=(0, 1), ylim=(0, 1))
axs[1, 0].axis('equal')
axs[1, 0].set_title("NAIL", fontsize=fontsize)

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
            dist_task = interaction_agent.task_models[task].get_dist(q_task)

            q_interactions = [interaction_agent.interaction_models[impact_task].get_q(state, 0, 1)]

            q_values = [q_task] + [q for q in q_interactions]

            dist_interactions = [interaction_agent.interaction_models[impact_task].get_dist(q_interactions[0])]
            dists = [dist_task.cpu().numpy().flatten()] + [d.cpu().numpy().flatten() for d in dist_interactions]
            entropies = [1 - interaction_agent.calc_entropy(inter) for inter in dists]
            r_ent = [1 - interaction_agent.calc_relative_entropy(dist, dists[0]) for dist in dists]
            support = len(dists[0])
            linear_entropies = [np.power(support, ent) / support for ent in entropies]
            linear_r_ent = [np.power(support, ent) / support for ent in r_ent]
            q_weights = [torch.sum(q)/torch.sum(sum(q_values)) for q in q_values]
            ent_weights = torch.Tensor([l_ent * l_r_ent / q_weight for l_ent, l_r_ent, q_weight in zip(linear_entropies, linear_r_ent, q_weights)])
            weights = ent_weights / sum(ent_weights)
            comb_q = []
            for q, w in zip(q_values, weights):
                comb_q.append(q * w.item())
            combined_q = torch.Tensor(sum(comb_q))
            combined_v = interaction_agent.task_models[task].get_v(combined_q)

            dist = torch.exp((combined_q - combined_v) / interaction_agent.task_alpha)
            dist /= torch.sum(dist)

            u[a, b] = dist.data[0, 1].item() - dist.data[0, 3].item()
            v[a, b] = dist.data[0, 2].item() - dist.data[0, 4].item()


# Quiver-Diagramm erstellen
q4 = axs[1, 1].quiver(x, y, u, v, angles='xy')
axs[1, 1].axis([0, 1, 0, 1])
axs[1, 1].invert_yaxis()
# axs[1, 1].quiverkey(q4, X=0.0, Y=1.0, U=10,
#              label='quiver(X, Y, U, V), invert_yaxis()', labelpos='E')
# axs[1, 1].margins(y=0)
# axs[1, 1].margins(x=0)
# ax.set(xlim=(0, 1), ylim=(0, 1))
axs[1, 1].axis('equal')
axs[1, 1].set_title("AAIL", fontsize=fontsize)

plt.savefig("plots/quiver_all.svg")

# Quiver-Diagramm anzeigen
plt.show()
