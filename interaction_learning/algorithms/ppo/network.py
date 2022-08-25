import torch
from torch import nn


class PPONetwork(nn.Module):

    def __init__(self, in_dim, out_dim, device="cpu"):
        super(PPONetwork, self).__init__()
        self._hidden_layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.device = device
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(128, out_dim))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(128, 1))
        self._output = None

    def forward(self, obs):
        symbolic_input: torch.Tensor = obs
        sym = symbolic_input.permute(0, 3, 1, 2)
        self._output = self._hidden_layers(sym)
        logits = self._logits(self._output)
        return logits

    def ego_value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._ego_value_branch(self._output).squeeze(1)

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # print('state : ', state)
        with torch.no_grad():
            q = self.forward(state)
            v = self.get_value(q).squeeze()
            # print('q & v', q, v)
            dist = torch.exp((q - v) / self.alpha)
            # print(dist)
            dist = dist / torch.sum(dist)
            # print(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()