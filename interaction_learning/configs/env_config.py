import torch

goal_dim = 2
action_space_shape = (1, )
num_actions = 5
self_observation_dim_type1 = 4
self_observation_dim_type2 = 5
landmark_observation_dim = 2
global_state_dim = 4
max_steps = 50

possible_actions = torch.Tensor([0, 1, 2, 3, 4])
prob_random = 0.2
action_mapping = {0: "no-op", 1: "up", 2: "down", 3: "left", 4: "right"}


