from interaction_learning.algorithms.interaction_framework.util.memory import Memory
from interaction_learning.algorithms.interaction_framework.util.soft_agent import SoftAgent
from interaction_learning.algorithms.interaction_framework.util.soft_interaction_agent import SoftInteractionAgent
import torch
from torch.distributions import Categorical

import pickle


class ParticleInteractionAgent:

    LEARN_INTERACTION = "learn_interaction"
    LEARN_TASK = "learn_task"
    INFERENCE = "inference"

    def __init__(self, obs_dim, n_actions, task_alpha, impact_alpha, batch_size, gamma, target_update_interval,
                 memory_size, device):
        self.task_models = {}
        self.interaction_models = {}
        self.memory_size = memory_size
        self.memory = Memory(memory_size)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.task_alpha = task_alpha
        self.impact_alpha = impact_alpha
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.device = device
        self.current_active_tasks = []
        self.mode = self.LEARN_TASK
        self.learn_step_counter = 0

    def load_task_model(self, task, path):
        pass

    def save_agent(self, base_path, clear_memory=True):
        if clear_memory:
            self.memory.clear()
            self.learn_step_counter = 0
        with open(base_path, "wb") as output_file:
            pickle.dump(self, output_file)

    def get_active_task_model(self):
        return self.task_models[self.get_active_task()]

    def get_active_task(self):
        return [task for task in self.current_active_tasks if "t" in task][0]

    def add_task(self, task):
        new_model = SoftAgent(self.obs_dim, self.n_actions, self.task_alpha, self.batch_size, self.gamma,
                              self.target_update_interval, self.device)
        self.task_models[task] = new_model

    def add_interaction(self, interaction):
        new_model = SoftInteractionAgent(self.obs_dim, self.n_actions, self.impact_alpha, self.batch_size, self.gamma,
                                         self.target_update_interval, self.device)
        self.interaction_models[interaction] = new_model

    def switch_active_tasks(self, new_active_tasks):
        self.current_active_tasks = new_active_tasks
        self.memory.clear()
        self.learn_step_counter = 0

    def switch_mode(self, new_mode):
        assert new_mode in [self.LEARN_TASK, self.LEARN_INTERACTION, self.INFERENCE]
        self.mode = new_mode
        self.memory.clear()
        self.learn_step_counter = 0

    def select_action(self, state, self_agent_num=0, other_agent_num=1):
        active_task = [task for task in self.current_active_tasks if "t" in task]
        active_interactions = [interaction for interaction in self.current_active_tasks if "i" in interaction]
        if active_task and not active_interactions:
            action = self.task_models[active_task[0]].select_action(state)
        elif not active_task and len(active_interactions) == 1:
            action = self.interaction_models[active_interactions[0]].select_action(state, self_agent_num,
                                                                                   other_agent_num)
        elif active_task and active_interactions:
            action = self.combine_task_actions(state, active_task, active_interactions, self_agent_num,
                                               other_agent_num)
        else:
            raise Exception("Unexpected Active Task or Interaction Combination. Maybe none is selected?")
        return action

    def add_transition(self, state, next_state, action, reward, done):
        self.memory.add((state, next_state, action, reward, done))

    def learn_step(self, other_agent_num=1, self_agent_num=0, other_agent_model=None):
        self.learn_step_counter += 1
        if self.learn_step_counter < 250:
            return
        if self.mode is self.LEARN_TASK:
            active_task = [task for task in self.current_active_tasks if "t" in task][0]
            self.task_models[active_task].learn(self.memory)
        elif self.mode is self.LEARN_INTERACTION:
            active_task = [task for task in self.current_active_tasks if "i" in task][0]
            self.interaction_models[active_task].learn(self.memory, other_agent_num, self_agent_num, other_agent_model)

    def combine_task_actions(self, state, active_task, active_interactions, self_agent_num, other_agent_nums):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_task = self.task_models[active_task[0]].get_q(state)
            v_task = self.task_models[active_task[0]].get_v(state)
            q_interactions = [self.interaction_models[inter].get_q(state, self_agent_num, o_num)
                              for inter, o_num in zip(active_interactions, other_agent_nums)]
            v_interactions = [self.interaction_models[inter].get_v(state) for inter in active_interactions]

            combined_q = (q_task + sum(q_interactions)) / (1 + len(q_interactions))
            combined_v = (v_task + sum(v_interactions)) / (1 + len(v_interactions))

            dist = torch.exp((combined_q - combined_v) / self.task_alpha)
            c = Categorical(dist)
            a = c.sample()
        return a.item()
