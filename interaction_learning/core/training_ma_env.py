from typing import Tuple, Dict
from interaction_learning.plotting.progress_plot import plot
import torch
import numpy as np


def select_action(dqn_agent, state: np.ndarray) -> np.ndarray:
    """Select an action from the input state."""
    # NoisyNet: no epsilon greedy action selection
    selected_action = dqn_agent.dqn(
        torch.FloatTensor(state).to(dqn_agent.device)
    ).argmax()
    selected_action = selected_action.detach().cpu().numpy()

    if not dqn_agent.is_test:
        dqn_agent.transition = [state, selected_action]

    return selected_action


def step(dqn_agent, env, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    """Take an action and return the response of the env."""
    next_state, reward, done, _ = env.step(action)

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


def train(dqn_agent, env, num_steps: int, plotting_interval: int = 200):
    """Train the agent."""
    dqn_agent.is_test = False

    state = env.reset()
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
            state = env.reset()
            scores.append(score)
            score = 0

        # if training is ready
        if len(dqn_agent.memory) >= dqn_agent.batch_size:
            loss = dqn_agent.update_model()
            losses.append(loss)
            update_cnt += 1

            # if hard update is needed
            if update_cnt % dqn_agent.target_update == 0:
                dqn_agent._target_hard_update()

        # plotting
        if frame_idx % plotting_interval == 0:
            plot(frame_idx, scores, losses)

    env.close()
