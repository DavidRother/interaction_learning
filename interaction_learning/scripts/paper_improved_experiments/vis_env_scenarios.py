from partigames.environment.zoo_env import parallel_env
import numpy as np
from time import sleep


max_steps = 1000
ghost_agents = 0
render = True


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e3"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot()


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e3"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot()


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e3"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot()


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e3"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot()
