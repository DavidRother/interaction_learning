import torch


class SoftDQNAgent:

    def __init__(self):
        self.optimizer = optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    def select_action(self, state):
        b = random.random()
        if self.epsilon > b:
            selected_action = self.action_space.sample()
        else:
            with torch.no_grad():
                selected_action = self.dqn(torch.Tensor(state).to(self.device)).argmax()
                selected_action = selected_action.detach().cpu().numpy()

        return selected_action

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

    def update_model(self) -> torch.Tensor:
        pass

