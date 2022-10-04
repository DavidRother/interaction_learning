import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from collections import deque
import random
from torch.distributions import Categorical
import gym
import numpy as np


class SoftQNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, alpha, device="cpu"):
        super(SoftQNetwork, self).__init__()
        self.alpha = alpha
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_layer = nn.Sequential(nn.Linear(128, out_dim))
        self.device = device

    def forward(self, x):
        feature = self.feature_layer(x)
        if torch.isnan(feature).any().cpu().item():
            print("how")
        dist = self.value_layer(feature)
        if torch.isnan(dist).any().cpu().item():
            print("dist how")
        return dist

    def get_value(self, q_value):
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value / self.alpha), dim=1, keepdim=True))
        return v

    def select_action(self, state, greedy=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # print('state : ', state)
        with torch.no_grad():
            q = self.forward(state)
            v = self.get_value(q).squeeze()
            # print('q & v', q, v)
            dist = torch.exp((q - v) / self.alpha)
            # print(dist)
            dist = dist / torch.sum(dist)
            if greedy:
                a = np.argmax(dist)
            else:
                # print(dist)
                c = Categorical(dist)
                a = c.sample()
        return a.item()
