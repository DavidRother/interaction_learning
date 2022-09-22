import copy

from ray.rllib.algorithms.sac import SAC
from ray.tune.registry import register_env
from partigames.environment.gym_env import GymPartiEnvironment
import numpy as np

env_config = {
    "agent_position_generator": lambda: [np.asarray([0.05, np.random.uniform(0.01, 0.99, 1).item()])],
    "agent_reward": "x",
    "max_steps": 1000,
    "ghost_agents": 0,
    "render": False
}


def env_creator(env_conf):
    return GymPartiEnvironment(**env_conf)  # return an env instance


register_env("my_env", env_creator)

config = {
    "env": "my_env",
    "env_config": env_config,
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

algo.restore("./agents_rllib/sac_agent_test.mod")

test_config = copy.deepcopy(env_config)
test_config["render"] = True
test_env = env_creator(test_config)

done = False
obs = test_env.reset()
episode_reward = 0
test_env.render()

while not done:
    action = algo.compute_action(obs)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward
    test_env.render()

print(f"Episode Reward: {episode_reward}")
