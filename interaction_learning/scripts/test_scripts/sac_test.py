import os
import yaml
from datetime import datetime

from interaction_learning.algorithms.sacd.env import make_pytorch_env
from interaction_learning.algorithms.sacd.agent import SacdAgent, SharedSacdAgent


config_path = os.path.join('../../algorithms/sacd/config', 'sacd.yaml')
env_id = 'MsPacmanNoFrameskip-v4'
shared = False
cuda = False
seed = 0

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Create environments.
env = make_pytorch_env(env_id, clip_rewards=False)
test_env = make_pytorch_env(
    env_id, episode_life=False, clip_rewards=False)

# Specify the directory to log.
name = config_path.split('/')[-1].rstrip('.yaml')
if shared:
    name = 'shared-' + name
time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join(
    'logs', env_id, f'{name}-seed{seed}-{time}')

# Create the agent.
Agent = SacdAgent if not shared else SharedSacdAgent
agent = Agent(
    env=env, test_env=test_env, log_dir=log_dir, cuda=cuda,
    seed=seed, **config)
agent.run()
