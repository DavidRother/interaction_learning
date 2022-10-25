import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.optim as optim
import pickle
from interaction_learning.agents.generic_model_ppo import GenericModel
from interaction_learning.core.evaluation import evaluate, DummyAgent
from partigames.environment.zoo_env import parallel_env
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import torch
import tqdm
import pickle
from interaction_learning.utils.episode import collect_experience
from interaction_learning.algorithms.ppo.buffer import MultiAgentBuffer, REWARDS
from interaction_learning.algorithms.ppo.postprocessing import postprocess
from interaction_learning.algorithms.ppo.ppo_loss import ppo_surrogate_loss
from interaction_learning.agents.foraging_model_ppo import ForagingModel
from interaction_learning.algorithms.interaction_framework.particle_interaction_agent import ParticleInteractionAgent
from time import sleep


device = torch.device("cpu")

# 0 is do nothing 1 is move right 2 is down 3 is left 4 is up

agent_position_generator = lambda: [np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
                                    np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])]


agent_reward = [f"ta1", "tb1"]
max_steps = 1000
ghost_agents = 0
render = True
num_agents = 2

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                   agent_reward=agent_reward, max_steps=max_steps, ghost_agents=ghost_agents, render=render)
# temperature = torch.Tensor(alpha)
alpha = 0.1
impact_alpha = 0.3
with open("impact_learner/all_ego_and_impact_task.agent", "rb") as input_file:
    interaction_agent = pickle.load(input_file)

with open(f"impact_learner/all_ego_task.agent", "rb") as input_file:
    other_agent = pickle.load(input_file)

with open(f"impact_learner/joint_learner_mis_goals.agent", "rb") as input_file:
    joint_agent = pickle.load(input_file)

interaction_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
active_tasks = [agent_reward[0], agent_reward[1].replace("t", "i")]
interaction_agent.switch_active_tasks(active_tasks)
other_agent.switch_mode(ParticleInteractionAgent.INFERENCE)
other_agent.switch_active_tasks([agent_reward[1]])

state = env.reset()
episode_reward = 0
for time_steps in range(max_steps):
    obs = state["player_0"]
    obs[4] = 0.0
    obs[5] = 0.0
    obs[6] = 0.0
    obs[7] = 0.0
    action = interaction_agent.select_action(state["player_0"], 0, [1])
    action_2 = other_agent.select_action(state["player_1"])
    actions = {"player_0": action, "player_1": action_2}
    next_state, reward, done, _ = env.step(actions)
    episode_reward += reward["player_0"]
    print(reward)

    env.render(mode="human")
    sleep(0.1)

    if any(done.values()):
        break

    state = next_state

print(episode_reward)

