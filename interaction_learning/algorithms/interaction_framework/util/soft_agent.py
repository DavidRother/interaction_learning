from interaction_learning.algorithms.interaction_framework.util.soft_dqn_network import SoftQNetwork
import torch
import torch.nn.functional as F
import numpy as np


class SoftAgent:

    def __init__(self, obs_dim, n_actions, alpha, batch_size, gamma, target_update_interval, device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.device = device
        self.model = SoftQNetwork(self.obs_dim, self.n_actions, self.alpha, device=self.device).to(self.device)
        self.target_model = SoftQNetwork(self.obs_dim, self.n_actions, self.alpha, device=self.device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.batch_size = batch_size
        self.learn_steps = 0
        self.gamma = gamma
        self.target_update_interval = target_update_interval

    def learn(self, memory):
        if self.learn_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        batch = memory.sample(self.batch_size, False)
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        batch_state = torch.FloatTensor(np.asarray(batch_state)).to(self.device)
        batch_next_state = torch.FloatTensor(np.asarray(batch_next_state)).to(self.device)
        batch_action = torch.FloatTensor(np.asarray(batch_action)).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(np.asarray(batch_reward)).unsqueeze(1).to(self.device)
        batch_done = torch.FloatTensor(np.asarray(batch_done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_q = self.target_model(batch_next_state)
            next_v = self.target_model.get_value(next_q)
            y = batch_reward + (1 - batch_done) * self.gamma * next_v

        loss = F.mse_loss(self.model(batch_state).gather(1, batch_action.long()), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        return loss.cpu().item()

    def select_action(self, state):
        return self.model.select_action(state)

    def get_q(self, state):
        return self.model(state)

    def get_v(self, q):
        return self.model.get_value(q)

    def get_dist(self, q):
        return self.model.get_dist(q)

