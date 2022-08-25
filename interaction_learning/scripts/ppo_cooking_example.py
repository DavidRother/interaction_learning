import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.optim as optim
import pickle
from gym_cooking.environment import cooking_zoo
from gym_cooking.cooking_book.recipe_drawer import RECIPES
from interaction_learning.utils.episode import collect_experience
from interaction_learning.algorithms.ppo.buffer import MultiAgentBuffer, REWARDS
from interaction_learning.algorithms.ppo.postprocessing import postprocess
from interaction_learning.algorithms.ppo.ppo_loss import ppo_surrogate_loss
from interaction_learning.agents.cooking_model import CookingModel


# environment
n_agents = 1
num_humans = 0
render = False

level = 'open_room_interaction2'
record = False
max_num_timesteps = 100

goal_encodings = {name: recipe().goal_encoding for name, recipe in RECIPES.items()}

recipes_to_learn = ["TomatoLettuceSalad", "CarrotBanana", "CucumberOnion", "AppleWatermelon"]
recipes = ["TomatoLettuceSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"],
                               ghost_agents=1)

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]
obs_dim = obs_space.shape[0]

num_agents = 1
training_iterations = 500000
training_steps = 400
num_batches = 1
num_epochs = 4
lr = 1e-4

agent_1_config = {"prosocial_level": 0.0, "update_prosocial_level": False, "use_prosocial_head": False}

agent_configs = {"player_0": agent_1_config}

# initialize agents
agents = {f"player_{idx}": CookingModel(obs_dim, action_space.n) for idx in range(num_agents)}
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
        with open(f'agents/ppo_agent_checkpoint_{training_iteration}', "wb") as output_file:
            pickle.dump(agents["player_0"], output_file)

with open(f'agents/ppo_agent', "wb") as output_file:
    pickle.dump(agents["player_0"], output_file)
