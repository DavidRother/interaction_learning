import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.optim as optim
import pickle
from lbforaging.foraging import zoo_environment
from interaction_learning.utils.episode import collect_experience
from interaction_learning.algorithms.ppo.buffer import MultiAgentBuffer, REWARDS
from interaction_learning.algorithms.ppo.postprocessing import postprocess
from interaction_learning.algorithms.ppo.ppo_loss import ppo_surrogate_loss
from interaction_learning.agents.foraging_model_ppo import ForagingModel


# environment
players = 1
max_player_level = 3
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
normalize_reward = True
grid_observation = False
penalty = 0.0

# env = gym.make("Foraging-8x8-2p-2f-v2")
env = zoo_environment.parallel_env(players=players, max_player_level=max_player_level, field_size=field_size,
                                   max_food=max_food, sight=sight, max_episode_steps=max_episode_steps,
                                   force_coop=force_coop, normalize_reward=normalize_reward,
                                   grid_observation=grid_observation, penalty=penalty)

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]
obs_dim = obs_space.shape[0]

num_agents = 1
training_iterations = 100000
training_steps = 400
num_batches = 1
num_epochs = 4
lr = 1e-4

agent_1_config = {"prosocial_level": 0.0, "update_prosocial_level": False, "use_prosocial_head": False}

agent_configs = {"player_0": agent_1_config}

# initialize agents
agents = {f"player_{idx}": ForagingModel() for idx in range(num_agents)}
optimizer = {f"player_{idx}": optim.Adam(agents[f"player_{idx}"].parameters(), lr=lr) for idx in range(num_agents)}


# initialize training loop
buffer = MultiAgentBuffer(training_steps, num_agents)

for training_iteration in range(training_iterations):
    episode_stats = collect_experience(env, buffer, agents, training_steps)
    postprocess(agents, buffer, agent_configs)

    rewards = {player: sum(buffer.buffer[player][REWARDS]) for player in agents}

    for epoch in range(num_epochs):
        for player in agents:
            for batch in buffer.buffer[player].build_batches(num_batches):
                loss, stats = ppo_surrogate_loss(agents[player], buffer.buffer[player], agent_configs[player])

                optimizer[player].zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                optimizer[player].step()

        # print(player, stats)
    print(episode_stats)
    buffer.reset()

    if (training_iteration % 50000) == 49999:
        with open(f'agents/ppo_agent_checkpoint_{training_iteration}.pickle', "wb") as output_file:
            pickle.dump(agents["player_0"], output_file)

with open(f'agents/ppo_agent_tomato_lettuce_salad.pickle', "wb") as output_file:
    pickle.dump(agents["player_0"], output_file)
