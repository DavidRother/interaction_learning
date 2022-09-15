from ray.rllib.algorithms.sac import SAC

config = {
    "env": "CartPole-v0",
    "num_workers": 0,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_config": {
        "render_env": True,
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

