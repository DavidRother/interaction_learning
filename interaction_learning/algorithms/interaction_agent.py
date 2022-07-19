from interaction_learning.utils.replay_buffer import ReplayBuffer
from interaction_learning.utils.priorotized_replay_buffer import PrioritizedReplayBuffer
from interaction_learning.algorithms.rainbow_network import Network
from interaction_learning.utils.struct_conversion import convert_numpy_obs_to_torch_dict
from interaction_learning.algorithms.rainbow.agent import DQNAgent


class InteractionAgent:

    def __init__(self, tom_model, impact_model, obs_space, action_space, batch_size, target_update,
                 initial_mem_requirement, obs_dim, memory_size, alpha, n_step, gamma):
        self.agents = {}
        self.tom_model = tom_model
        self.impact_model = impact_model

        self.obs_space = obs_space
        self.action_space = action_space
        self.target_update = target_update
        self.initial_mem_requirement = initial_mem_requirement
        self.batch_size = batch_size

        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

    def add_new_goal(self, goal):
        new_agent = DQNAgent(self.obs_space, self.action_space, self.batch_size, self.target_update,
                             self.initial_mem_requirement)
        self.agents[goal] = new_agent

    def train_interaction(self, goal):
        pass

    def select_multi_agent_action(self, goal, state):
        self_action_evaluations = self.agents[goal].recieve_action_evaluations(state)
        other_agent_state = self.transform_state_perspective(state)
        other_agent_goal = self.tom_model(other_agent_state)
        other_agent_action_evaluations = self.agents[other_agent_goal].recieve_action_evaluations(other_agent_state)
        final_action_evaluations = self.combine_action_evaluations(self_action_evaluations,
                                                                   other_agent_action_evaluations)
        action = self.select_action(final_action_evaluations)
        return action

    def transform_state_perspective(self, state):
        return state

    def combine_action_evaluations(self, self_action_evaluations, other_agent_action_evaluations):
        return self_action_evaluations

    def select_action(self, action_evaluations):
        return 0


