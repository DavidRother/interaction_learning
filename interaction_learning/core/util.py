import copy
import random
import numpy as np
import torch
import math
from math import pi


def make_deterministic(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


class AgentPositionGenerator:

    def __init__(self, num_pos, num_agents=2, x_min=0, x_max=1, y_min=0, y_max=1):
        self.agent_positions = [[np.asarray([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
                                 for i in range(num_agents)]
                                for _ in range(num_pos)]
        self.pointer = 0

    def __call__(self, *args, **kwargs):
        pos = self.agent_positions[self.pointer]
        self.pointer += 1
        return copy.deepcopy(pos)

    def reset(self):
        self.pointer = 0


class AgentPositionGenerator2:

    def __init__(self, num_pos, pos_constraints=None):
        self.agent_positions = [[np.asarray([np.random.uniform(pos[0], pos[1]), np.random.uniform(pos[2], pos[3])])
                                 for pos in pos_constraints] for _ in range(num_pos)]
        self.pointer = 0

    def __call__(self, *args, **kwargs):
        pos = self.agent_positions[self.pointer]
        self.pointer += 1
        return copy.deepcopy(pos)

    def reset(self):
        self.pointer = 0


# class AgentPositionGeneratorCircle:
#
#     def __init__(self, num_pos, num_agents=2, position_encodings=None):
#         self.agent_positions = [[] for enc in position_encodings] for _ in range(num_pos)]
#         self.pointer = 0
#
#     def __call__(self, *args, **kwargs):
#         pos = self.agent_positions[self.pointer]
#         self.pointer += 1
#         return copy.deepcopy(pos)
#
#     def reset(self):
#         self.pointer = 0
#
#     @staticmethod
#     def points_on_circumference(center=(0, 0), r=50, n=100):
#         return [(center[0] + (math.cos(2 * pi / n * x) * r),  center[1] + (math.sin(2 * pi / n * x) * r))
#                 for x in range(0, n + 1)]

