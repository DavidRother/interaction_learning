import torch
from torch import nn


class FeatureNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        nn.Module.__init__(self)

        self._hidden_layers = nn.Sequential(nn.Conv2d(51, 64, (3, 3), stride=2, padding=1),
                                            nn.Tanh(),
                                            nn.Conv2d(64, 64, (3, 3), padding=1),
                                            nn.Tanh(),
                                            nn.Flatten(),
                                            nn.Linear(1024, 256),
                                            nn.Tanh(),
                                            nn.Linear(256, output_shape[0]),
                                            )
        self._output = None

    def forward(self, obs):
        symbolic_input: torch.Tensor = obs["symbolic_observation"]
        sym = symbolic_input.permute(0, 3, 1, 2)
        self._output = self._hidden_layers(sym)
        return self._output
