import torch
from torch import nn


class ImageModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._hidden_layers = nn.Sequential(nn.Conv2d(3, 32, (3, 3), stride=2, padding=1),
                                            nn.ELU(),
                                            nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
                                            nn.ELU(),
                                            nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
                                            nn.ELU(),
                                            nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
                                            nn.ELU(),
                                            nn.Flatten(),
                                            nn.Linear(288, 64),
                                            )
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(64, 5))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(64, 1))
        self._ego_value_branch = nn.Sequential(nn.Tanh(), nn.Linear(64, 1))
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
