from interaction_learning.algorithms.interaction_framework.util.soft_dqn_network import SoftQNetwork
import torch
import torch.nn.functional as F
import numpy as np


class SoftInteractionAgent:

    def __init__(self, obs_dim, n_actions, alpha, batch_size, gamma, target_update_interval, device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.device = device
        self.model = SoftQNetwork(self.obs_dim * 2, self.n_actions, self.alpha, device=self.device).to(self.device)
        self.target_model = SoftQNetwork(self.obs_dim * 2, self.n_actions, self.alpha, device=self.device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.batch_size = batch_size
        self.learn_steps = 0
        self.gamma = gamma
        self.target_update_interval = target_update_interval

    def learn(self, memory, other_agent_num, self_agent_num, other_agent_model):
        if self.learn_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        batch = memory.sample(self.batch_size, False)
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        batch_state = torch.FloatTensor(np.asarray(batch_state)).to(self.device)
        batch_next_state = torch.FloatTensor(np.asarray(batch_next_state)).to(self.device)
        batch_action = torch.FloatTensor(np.asarray(batch_action)).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(np.asarray(batch_reward)).unsqueeze(1).to(self.device)
        batch_done = torch.FloatTensor(np.asarray(batch_done)).unsqueeze(1).to(self.device)

        other_agent_batch_state = self.transform_state_pov(batch_state, self_agent_num, other_agent_num)
        other_agent_next_batch_state = self.transform_state_pov(batch_next_state, self_agent_num, other_agent_num)

        with torch.no_grad():
            current_q = other_agent_model.get_q(other_agent_batch_state)
            current_v = other_agent_model.get_v(current_q)
            next_q = other_agent_model.get_q(other_agent_next_batch_state)
            next_v = other_agent_model.get_v(next_q)
            impact_batch_next_state = torch.cat([batch_next_state, other_agent_next_batch_state], dim=1)
            future_impact_q = self.model(impact_batch_next_state)
            future_impact_value = self.model.get_value(future_impact_q)
            r = next_v + batch_reward - current_v
            y = r + (1 - batch_done) * self.gamma * future_impact_value

        impact_batch_state = torch.cat([batch_state, other_agent_batch_state], dim=1)
        loss = F.mse_loss(self.model(impact_batch_state).gather(1, batch_action.long()), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        return loss.cpu().item()

    def select_action(self, state, self_agent_num, other_agent_num):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self_state = state
        other_state = self.transform_state_pov(state, self_agent_num, other_agent_num)
        impact_state = torch.cat([self_state, other_state], dim=1)
        return self.model.select_action(impact_state)

    def get_q(self, state, self_agent_num, other_agent_num):
        self_state = state
        other_state = self.transform_state_pov(state, self_agent_num, other_agent_num)
        impact_state = torch.cat([self_state, other_state], dim=1)
        return self.model(impact_state)

    def get_v(self, q):
        return self.model.get_value(q)

    def transform_state_pov(self, batch_state, self_agent_num, other_agent_num):
        if isinstance(batch_state, np.ndarray):
            batch_state = torch.FloatTensor(batch_state).unsqueeze(0).to(self.device)
        total_num = batch_state.shape[1]
        oa_idx_inc = int(other_agent_num < self_agent_num)
        column_order = list(range((other_agent_num + oa_idx_inc) * 4,
                                  (other_agent_num + oa_idx_inc) * 4 + 4)) + list(range(4, total_num))
        idx_inc = int(self_agent_num < other_agent_num)
        column_order[(self_agent_num + idx_inc) * 4:(self_agent_num + idx_inc) * 4 + 4] = list(range(4))
        new_batch_state = torch.index_select(batch_state, 1, torch.LongTensor(column_order))
        return new_batch_state


