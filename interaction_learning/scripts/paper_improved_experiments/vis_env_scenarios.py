from partigames.environment.zoo_env import parallel_env
import numpy as np
from time import sleep


max_steps = 1000
ghost_agents = 0
render = True

# aligned_task_1 = ["ta", "te", "tb", "tc", "td"]
# impact_task_1 = ["ie0", "ia1", "id2", "ic4", "ib3"]
# aligned_task_2 = ["te0", "ta1", "td2", "tc4", "tb3"]


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e0"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot(path="./plots/screenshot_aligned.png")


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a", "e0"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot(path="./plots/screenshot_aligned1.png")


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["e", "a1"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot(path="./plots/screenshot_aligned2.png")


num_agents = 2
agent_position_generator = lambda: [np.asarray([0.367, 0.395]), np.asarray([0.59, 0.64])]
agent_reward = ["a3", "c4"]

env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator, agent_reward=agent_reward,
                   max_steps=max_steps, ghost_agents=ghost_agents, render=render)

obs = env.reset()

env.render(mode="human")
env.unwrapped.screenshot(path="./plots/screenshot_not_aligned.png")
