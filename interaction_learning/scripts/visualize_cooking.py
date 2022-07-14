import pickle
from interaction_learning.utils.struct_conversion import convert_dict_to_numpy, convert_numpy_obs_to_torch_dict
import torch
import numpy as np
from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game.game import Game


# environment
n_agents = 1
num_humans = 0
render = True

level = 'open_room_salad'
seed = 123463
record = False
max_num_timesteps = 100
recipes = ["TomatoSalad"]
action_scheme = "scheme3"

env_id = "CookingZoo-v0"
env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_num_timesteps,
                               recipes=recipes, action_scheme=action_scheme, obs_spaces=["feature_vector"])


with open(r"agent4_7x7_tomato_salad.pickle", "rb") as output_file:
    agent = pickle.load(output_file)


def select_action(dqn_agent, state: np.ndarray) -> np.ndarray:
    """Select an action from the input state."""
    with torch.no_grad():
        selected_action = dqn_agent.dqn(torch.Tensor(state).to(dqn_agent.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
    return selected_action


class CookingAgent:

    def get_action(self, observation) -> int:
        return select_action(agent, observation).item()


cooking_agent = CookingAgent()

game = Game(env, num_humans, [cooking_agent], max_num_timesteps, render=True)
store = game.on_execute_ai_only_with_delay(0.5)

print("done")
