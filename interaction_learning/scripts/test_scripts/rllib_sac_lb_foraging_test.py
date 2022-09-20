from ray.rllib.algorithms.sac import SAC
from ray.tune.registry import register_env
from lbforaging.foraging import environment

env_config = {
    "players": 1,
    "max_player_level": 3,
    "field_size": (8, 8),
    "max_food": 2,
    "sight": 8,
    "max_episode_steps": 50,
    "force_coop": False,
    "normalize_reward": True,
    "grid_observation": False,
    "penalty": 0.0,
}


def env_creator(env_conf):
    return environment.ForagingEnv(**env_config)  # return an env instance


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

for _ in range(1000):
    stuff = algo.train()
    interest_keys = ["episode_reward_max", "episode_reward_min", "episode_reward_mean", "episode_len_mean",
                     "episodes_this_iter", "episodes_total", "num_agent_steps_sampled"]
    filtered_stuff = {k: stuff[k] for k in interest_keys}
    print(filtered_stuff)

algo.evaluate()

