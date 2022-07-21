import pickle
from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game.game import Game
import sys
import interaction_learning

sys.modules['interaction_learning'] = interaction_learning
sys.path.append('C:/Users/David/PycharmProjects/interaction_learning/interaction_learning')
# environment
n_agents = 1
num_humans = 0
render = True

level = 'open_room_salad'
seed = 123463
record = False
max_num_timesteps = 100
recipes = ["TomatoLettuceSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"])

with open(r"./agents/checkpoint_10_agent0_7x7_tomato_salad_test3.pickle", "rb") as output_file:
    agent = pickle.load(output_file)


class CookingAgent:

    def get_action(self, observation) -> int:
        return agent.select_action(observation)


cooking_agent = CookingAgent()

game = Game(env, num_humans, [cooking_agent], max_num_timesteps, render=True)
store = game.on_execute_ai_only_with_delay(0.5)

print("done")
