from interaction_learning.utils.replay_buffer import ReplayBuffer
from interaction_learning.utils.priorotized_replay_buffer import PrioritizedReplayBuffer
from interaction_learning.algorithms.soft_dqn.soft_dqn_agent import SoftDQNAgent


class SoftInteractionAgent:

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
        self.obs_dim = obs_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma

        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)
        self.training_required = False
        self.current_goal = None
        self.is_test = False
        self.interaction_learning = False
        self.interactive_mode = False

    def add_new_goal(self, goal):
        new_agent = SoftDQNAgent(self.obs_space, self.action_space, self.batch_size, self.target_update,
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
        action = self.select_proxy_action(final_action_evaluations)
        return action

    def transform_state_perspective(self, state):
        return state

    def combine_action_evaluations(self, self_action_evaluations, other_agent_action_evaluations):
        return self_action_evaluations

    def select_proxy_action(self, action_evaluations):
        return 0

    def store_transition(self, transition):
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*transition)
        # 1-step transition
        else:
            one_step_transition = transition

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)

        if len(self.memory) >= self.initial_mem_requirement:
            self.training_required = True

    def switch_active_goal(self, goal):
        if goal not in self.agents:
            raise Exception("Please only switch to goals that have already been added")
        self.current_goal = goal
        self.memory = PrioritizedReplayBuffer(self.obs_dim, self.memory_size, self.batch_size, alpha=self.alpha)
        self.memory_n = ReplayBuffer(self.obs_dim, self.memory_size, self.batch_size,
                                     n_step=self.n_step, gamma=self.gamma)
        self.training_required = False

    def select_action(self, state):
        if self.interactive_mode:
            agent = self.get_current_agent()
            return agent.select_action(state)
        else:
            agent = self.get_current_agent()
            return agent.select_action(state)

    def update_model(self):
        loss = self.agents[self.current_goal].update_model(self.memory, self.memory_n, self.n_step)
        return {"Loss": loss}

    def postprocess_step(self, fraction):
        agent = self.get_current_agent()
        agent.beta = agent.beta + fraction * (1.0 - agent.beta)

    def stats(self, identifier=""):
        agent = self.get_current_agent()
        return {f"Epsilon {identifier}": agent.epsilon}

    def params(self, identifier=""):
        agent = self.get_current_agent()
        return agent.params(identifier)

    def get_current_agent(self):
        return self.agents[self.current_goal]

