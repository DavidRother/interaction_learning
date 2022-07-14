from typing import Tuple
from interaction_learning.plotting.progress_plot import plot
from interaction_learning.utils.struct_conversion import convert_dict_to_numpy, convert_numpy_obs_to_torch_dict
import torch
import numpy as np
import random


def convert_state(state, device):
    torch_state = {}
    for k, v in state.items():
        torch_state[k] = torch.FloatTensor(v).unsqueeze(dim=0).to(device)
    return torch_state


def select_action(dqn_agent, state: np.ndarray) -> np.ndarray:
    """Select an action from the input state."""
    b = random.random()
    if dqn_agent.epsilon > b:
        selected_action = dqn_agent.action_space.sample()
        # print(f"random action {selected_action}")
    else:
        with torch.no_grad():
            # selected_action = dqn_agent.dqn(convert_numpy_obs_to_torch_dict(state, dqn_agent.device, batch=False)).argmax()
            selected_action = dqn_agent.dqn(torch.Tensor(state).to(dqn_agent.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        # print(f"network action {selected_action}")

    if not dqn_agent.is_test:
        dqn_agent.transition = [state, selected_action]

    return selected_action


def step(dqn_agent, env, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    """Take an action and return the response of the env."""
    action_dict = {"player_0": action}
    next_state, reward, done, _ = env.step(action_dict)
    next_state = convert_dict_to_numpy(next_state["player_0"])
    done = done["player_0"]
    reward = reward["player_0"]

    if not dqn_agent.is_test:
        dqn_agent.transition += [reward, next_state, done]

        # N-step transition
        if dqn_agent.use_n_step:
            one_step_transition = dqn_agent.memory_n.store(*dqn_agent.transition)
        # 1-step transition
        else:
            one_step_transition = dqn_agent.transition

        # add a single step transition
        if one_step_transition:
            dqn_agent.memory.store(*one_step_transition)

    return next_state, reward, done


def train(dqn_agent, env, num_steps: int, evalpy_config=None):
    """Train the agent."""
    dqn_agent.is_test = False

    state = env.reset()["player_0"]
    update_cnt = 0
    losses = []
    scores = []
    score = 0

    for frame_idx in range(1, num_steps + 1):
        action = select_action(dqn_agent, state)
        next_state, reward, done = step(dqn_agent, env, action)

        state = next_state
        score += reward

        # NoisyNet: removed decrease of epsilon

        # PER: increase beta
        fraction = min(frame_idx / num_steps, 1.0)
        dqn_agent.beta = dqn_agent.beta + fraction * (1.0 - dqn_agent.beta)

        # if episode ends
        if done:
            print(f"final episode score: {score} | epsilon: {dqn_agent.epsilon}")
            state = convert_dict_to_numpy(env.reset()["player_0"])
            scores.append(score)
            score = 0

        # if training is ready
        if len(dqn_agent.memory) >= dqn_agent.init_mem_requirement:
            loss = dqn_agent.update_model()
            losses.append(loss)
            update_cnt += 1

            # if hard update is needed
            if update_cnt % dqn_agent.target_update == 0:
                dqn_agent._target_hard_update()

        # plotting
        # if frame_idx % plotting_interval == 0:
        #     plot(frame_idx, scores, losses)

    env.close()
