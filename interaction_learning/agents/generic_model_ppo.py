import torch
from torch import nn
from torch.distributions.categorical import Categorical


class GenericModel(nn.Module):

    def __init__(self, in_dim, out_dim, device="cpu"):
        nn.Module.__init__(self)
        self.device = device
        self._hidden_layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(128, out_dim))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(128, 1))
        self._output = None

    def forward(self, obs):
        self._output = self._hidden_layers(obs)
        logits = self._logits(self._output)
        return logits

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

    def select_action(self, state, return_logits=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self(state).squeeze(1)
        logits_numpy = logits.detach().cpu().numpy()
        action = Categorical(logits=logits).sample().item()
        if return_logits:
            return action, logits_numpy
        else:
            return action

