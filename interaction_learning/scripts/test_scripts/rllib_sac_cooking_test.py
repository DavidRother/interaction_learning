from ray.rllib.algorithms.sac import SAC
from ray.tune.registry import register_env
from gym_cooking.environment.environment import GymCookingEnvironment

n_agents = 1
num_humans = 1
render = False

level = 'open_room_interaction2'
seed = 1
record = False
max_num_timesteps = 100
recipe = "CarrotBanana"
action_scheme = "scheme3"


env_config = {
    "level": level, "record": record, "max_steps": max_num_timesteps,
    "recipe": recipe, "action_scheme": action_scheme, "obs_spaces": ["feature_vector"],
    "ghost_agents": 1
}


def env_creator(env_conf):
    return GymCookingEnvironment(**env_config)  # return an env instance


register_env("my_env", env_creator)

config = {
    "env": "my_env",
    "num_workers": 0,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    },
    "evaluation_config": {
        "render_env": False,
    }
}

algo = SAC(config=config)

for _ in range(10000):
    stuff = algo.train()
    interest_keys = ["episode_reward_max", "episode_reward_min", "episode_reward_mean", "episode_len_mean",
                     "episodes_this_iter", "episodes_total", "num_agent_steps_sampled"]
    filtered_stuff = {k: stuff[k] for k in interest_keys}
    print(filtered_stuff)

algo.evaluate()

