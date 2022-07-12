import torch
import numpy as np


obs_type = np.dtype([("symbolic_observation", np.int32, (7, 7, 16)), ("agent_position", np.int32, (2, )),
                     ("goal_vector", np.int32, (11, ))])


def convert_dict_to_numpy(obs):
    return np.array(tuple([v for v in obs.values()]), dtype=obs_type)


def convert_numpy_obs_to_torch_dict(obs, device, batch=False):
    torch_state = {}
    for k in obs.dtype.names:
        if batch:
            torch_state[k] = torch.FloatTensor(obs[k]).to(device)
        else:
            torch_state[k] = torch.FloatTensor(obs[k]).unsqueeze(dim=0).to(device)
    return torch_state
