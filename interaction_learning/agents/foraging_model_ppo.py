import torch
from torch import nn


class ForagingModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self._hidden_layers = nn.Sequential(nn.Conv2d(13, 64, (3, 3), stride=2, padding=1),
                                            nn.Tanh(),
                                            nn.Conv2d(64, 64, (3, 3), padding=1),
                                            nn.Tanh(),
                                            nn.Flatten(),
                                            nn.Linear(1024, 256),
                                            nn.Tanh(),
                                            nn.Linear(256, 64),
                                            )
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(64, 6))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(64, 1))
        self._ego_value_branch = nn.Sequential(nn.Tanh(), nn.Linear(64, 1))
        self._output = None

    def forward(self, obs):
        symbolic_input: torch.Tensor = obs
        sym = symbolic_input.permute(0, 3, 1, 2)
        self._output = self._hidden_layers(sym)
        logits = self._logits(self._output)
        return logits

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def ego_value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._ego_value_branch(self._output).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())