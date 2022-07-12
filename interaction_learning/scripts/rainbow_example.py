from typing import Dict, List, Tuple
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from interaction_learning.algorithms.rainbow_agent import DQNAgent
from interaction_learning.core.training import train

# environment
env_id = "CartPole-v0"
env = gym.make(env_id)


seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
num_frames = 20000
memory_size = 10000
batch_size = 128
target_update = 100

# train
agent = DQNAgent(env, memory_size, batch_size, target_update)

train(agent, env, num_frames)

