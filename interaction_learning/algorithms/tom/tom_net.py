import torch
from torch import nn
import torch.nn.functional as F


class TomNet:

    def __init__(
            self,
            out_dim: int
    ):
        """Initialization."""
        super(TomNet, self).__init__()
        self.out_dim = out_dim

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 16, 128),
            nn.ReLU(),
        )

        self.comb_layer = nn.Sequential(
            nn.Linear(129, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x) -> torch.Tensor:
        """Forward method implementation."""
        state = x["obs"]
        action = x["action"]
        features = self.feature_layer(state)
        concat = torch.concat([features, action], dim=1)
        out = self.comb_layer(concat)

        return out



