import torch
from interaction_learning.utils.replay_buffer import ReplayBuffer
from interaction_learning.utils.priorotized_replay_buffer import PrioritizedReplayBuffer
from interaction_learning.algorithms.soft_dqn.soft_dqn_utils import SoftQNetwork
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class SoftDQNAgent:

    def __init__(self, obs_space, action_space, batch_size: int, target_update: int,
                 init_mem_requirement: int = 128, device="cpu", gamma: float = 0.99,
                 beta: float = 0.6, prior_eps: float = 1e-6):
        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        self.action_space = action_space
        self.init_mem_requirement = init_mem_requirement
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.device = torch.device("cpu")
        self.soft_dqn = SoftQNetwork(obs_dim, action_dim, device).to(device)
        self.target_soft_dqn = SoftQNetwork(obs_dim, action_dim, device).to(device)
        self.target_soft_dqn.load_state_dict(self.soft_dqn.state_dict())
        self.target_soft_dqn.eval()

        self.optimizer = torch.optim.Adam(self.soft_dqn.parameters(), lr=1e-4)

        self.beta = beta
        self.prior_eps = prior_eps
        # mode: train / test
        self.is_test = False

    def select_action(self, state):
        selected_action = self.soft_dqn.select_action(state)
        return selected_action

    def update_model(self, memory, n_step_memory=None, n_step=3) -> torch.Tensor:
        samples = memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        if n_step_memory:
            gamma = self.gamma ** n_step
            samples = n_step_memory.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.soft_dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)

        return loss.item()

    def target_hard_update(self):
        """Hard update: target <- local."""
        self.target_soft_dqn.load_state_dict(self.soft_dqn.state_dict())

    def params(self, identifier=""):
        params = {"init_mem_requirement": self.init_mem_requirement, "batch_size": self.batch_size,
                  "target_update": self.target_update, "gamma": self.gamma}
        if identifier:
            params = {f"{k} {identifier}": v for k, v in params.items()}
        return params

    def _compute_loss(self, samples, gamma):
        batch_state = torch.Tensor(samples["obs"]).to(self.device)
        batch_next_state = torch.Tensor(samples["next_obs"]).to(self.device)
        batch_action = torch.LongTensor(samples["acts"]).to(self.device)
        batch_reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        batch_done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        with torch.no_grad():
            next_q = self.target_soft_dqn(batch_next_state)
            next_v = self.target_soft_dqn.get_value(next_q)
            y = batch_reward + (1 - batch_done) * gamma * next_v

        q = self.soft_dqn(batch_state)
        actions = batch_action.long().unsqueeze(dim=-1)
        x = q.gather(1, actions)
        loss = F.mse_loss(x, y, reduction="none")
        return loss
