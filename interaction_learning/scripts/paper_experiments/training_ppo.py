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



x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
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

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]
obs_dim = env.observation_spaces["player_0"].shape[0]
n_actions = env.action_spaces["player_0"].n

training_steps = 1000
num_batches = 1
lr = 1e-4

agent_1_config = {"prosocial_level": 0.0, "update_prosocial_level": False, "use_prosocial_head": False}

agent_configs = {"player_0": agent_1_config}

# initialize agents
ppo_agent = GenericModel(obs_dim, n_actions)
ppo_optimizer = optim.Adam(ppo_agent.parameters(), lr=lr)

# initialize training loop
buffer = MultiAgentBuffer(training_steps, num_agents)

dummy_agent = DummyAgent()
ep_rews = {}
eval_scores = {}

collected_steps = 0
num_epochs = 200

for task in tasks:

    env = parallel_env(num_agents=num_agents, agent_position_generator=agent_position_generator,
                       agent_reward=[task, ""],
                       max_steps=max_steps, ghost_agents=ghost_agents, render=render)
    episode_reward = 0
    num_eval_runs = 100

    episode_rewards = []

    action_bag = []

    evaluation_scores = [evaluate(env, 10, [ppo_agent, dummy_agent])]

    for epoch in tqdm.tqdm(range(num_epochs)):
        state = env.reset()
        episode_reward = 0
        for time_steps in range(max_steps):
            action, logit = ppo_agent.select_action(state["player_0"], return_logits=True)
            actions = {"player_0": action, "player_1": 0}
            next_state, reward, done, info = env.step(actions)
            episode_reward += reward["player_0"]

            logits = {"player_0": logit, "player_1": logit}
            buffer.commit(state, actions, next_state, reward, done, info, logits, time_steps)

            if done["player_0"]:
                break

            state = next_state
            if render:
                env.render(mode="human")
                sleep(0.1)

            if collected_steps >= training_steps:
                collected_steps = 0
                postprocess(["player_0", "player_1"], buffer, agent_configs)
                for ep in range(4):
                    player = "player_0"
                    for batch in buffer.buffer[player].build_batches(num_batches):
                        loss, stats = ppo_surrogate_loss(ppo_agent, batch, agent_configs[player])

                        ppo_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                        ppo_optimizer.step()
                buffer.reset()
        if episode_reward > 0:
            print(f"Reward higher than anticipated: {action_bag}")
        action_bag = []
        episode_rewards.append(episode_reward)
        if epoch % 10 == 0:
            evaluation_scores.append(evaluate(env, 10, [ppo_agent, dummy_agent]))

    print(episode_rewards)
    ep_rews[task] = episode_rewards
    eval_scores[task] = evaluation_scores

    with open(f'ppo_learner/ppo_single_learner_{task}.agent', "wb") as output_file:
        pickle.dump(ppo_agent, output_file)

stats = {"ep_rews": ep_rews, "eval_scores": eval_scores}
with open("stats/training_ppo.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
