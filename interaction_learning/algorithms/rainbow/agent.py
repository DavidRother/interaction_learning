from typing import Dict, List, Tuple

import random
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from interaction_learning.algorithms.rainbow_network import Network
from interaction_learning.utils.struct_conversion import convert_numpy_obs_to_torch_dict


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            obs_space,
            action_space,
            batch_size: int,
            target_update: int,
            init_mem_requirement: int = 128,
            gamma: float = 0.99,
            # PER parameters
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            epsilon=0.4,
            epsilon_decay=0.000001,
            epsilon_min=0.2
    ):
        """Initialization.

        Args:
            obs_space
            action_space
            batch_size (int): batch size for sampling
            init_mem_requirement (int): initial number of samples to start learning
            target_update (int): period for target model's hard update
            gamma (float): discount factor
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
        """
        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        self.action_space = action_space
        self.init_mem_requirement = init_mem_requirement
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device("cpu")
        # print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state):
        b = random.random()
        if self.epsilon > b:
            selected_action = self.action_space.sample()
        else:
            with torch.no_grad():
                selected_action = self.compute_q_values(state).argmax()
                selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def compute_q_values(self, state):
        return self.dqn(torch.Tensor(state).to(self.device))

    def update_model(self, memory, n_step_memory=None, n_step=3) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if n_step_memory:
            gamma = self.gamma ** n_step
            samples = n_step_memory.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
        self.epsilon = max([self.epsilon_min, self.epsilon - self.epsilon_decay])

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.Tensor(samples["obs"]).to(device)
        next_state = torch.Tensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def params(self, identifier=""):
        params = {"init_mem_requirement": self.init_mem_requirement, "batch_size": self.batch_size,
                  "target_update": self.target_update, "gamma": self.gamma, "Start_epsilon": self.epsilon,
                  "epsilon_decay": self.epsilon_decay, "epsilon_min": self.epsilon_min, "v_min": self.v_min,
                  "v_max": self.v_max, "atom_size": self.atom_size}
        if identifier:
            params = {f"{k} {identifier}": v for k, v in params.items()}
        return params
