import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.optim as optim
import pickle
from gym_cooking.environment import cooking_zoo
from gym_cooking.cooking_book.recipe_drawer import RECIPES
from interaction_learning.algorithms.interaction_framework.cooking_interaction_agent import CookingInteractionAgent
from interaction_learning.core.evaluation import evaluate, DummyAgent
from time import sleep
import tqdm
import numpy as np


device = 'cpu'

# environment
n_agents = 1
num_humans = 0
render = False

level = 'open_room_interaction2'
record = False
max_steps = 100

goal_encodings = {name: recipe().goal_encoding for name, recipe in RECIPES.items()}

recipes_to_learn = ["TomatoLettuceSalad", "CarrotBanana", "CucumberOnion", "AppleWatermelon"]
recipes = ["CarrotBanana"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_steps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"],
                               ghost_agents=1)

obs_space, action_space = env.observation_spaces["player_0"], env.action_spaces["player_0"]
obs_dim = env.observation_spaces["player_0"].shape[0]
obs_dim_impact = obs_dim
n_actions = env.action_spaces["player_0"].n
task_alpha = 0.04
impact_alpha = 0.04
batch_size = 32
gamma = 0.99
target_update_interval = 1000
memory_size = 50000
action_alignment = False

num_epochs = 400

task = recipes[0]

interaction_agent = CookingInteractionAgent(obs_dim, obs_dim_impact, n_actions, task_alpha, impact_alpha,
                                            action_alignment, batch_size, gamma, target_update_interval,
                                            memory_size, device)

dummy_agent = DummyAgent()
ep_rews = {}
eval_scores = {}
ep_length_stats = {}


env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_steps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"],
                               ghost_agents=1)

interaction_agent.add_task(task)
interaction_agent.switch_active_tasks([task])

episode_reward = 0
num_eval_runs = 100

action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
action_dists = []

episode_rewards = []
episode_lengths = []
average_q = []

action_bag = []

evaluation_scores = [evaluate(env, 10, [interaction_agent])]

for epoch in tqdm.tqdm(range(num_epochs)):
    state = env.reset()
    episode_reward = 0
    ep_length = 0
    for time_steps in range(max_steps):
        action = interaction_agent.select_action(state["player_0"])
        actions = {"player_0": action, "player_1": 0}
        next_state, reward, done, info = env.step(actions)
        episode_reward += reward["player_0"]

        interaction_agent.add_transition(state["player_0"], next_state["player_0"], action,
                                         reward["player_0"], done["player_0"])

        action_distribution[action] += 1
        ep_length += 1

        if time_steps % 4 == 0:
            interaction_agent.learn_step()

        if info["player_0"]["done_once"]["player_0"]:
            dist = np.asarray(list(action_distribution.values())) / sum(action_distribution.values())
            action_dists.append(dist)
            action_distribution = {n: 0 for n in range(env.action_spaces["player_0"].n)}
            break

        state = next_state
        if render:
            env.render(mode="human")
            sleep(0.1)
    # if episode_reward > 1000:
    #     print(f"Reward higher than anticipated: {episode_reward}")
    action_bag = []
    episode_rewards.append(episode_reward)
    episode_lengths.append(ep_length)
    # print(f"Episode Rewards: {episode_reward} || Episode Length: {ep_length}")
    if epoch % 10 == 0:
        evaluation_scores.append(evaluate(env, 10, [interaction_agent]))
        # torch.save(onlineQNetwork.state_dict(), f'agent/sql{epoch}policy_y0dimpact')
        # print('Epoch {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
print(episode_rewards)
ep_rews[task] = episode_rewards
eval_scores[task] = evaluation_scores
ep_length_stats[task] = episode_lengths


stats = {"ep_rews": ep_rews, "eval_scores": eval_scores, "ep_lengths": ep_length_stats}
with open("stats/training_impact_learners.stats", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(stats, outp, pickle.HIGHEST_PROTOCOL)
interaction_agent.save_agent("impact_learner/all_ego_task.agent")