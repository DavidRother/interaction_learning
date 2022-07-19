from typing import Tuple
from interaction_learning.utils.struct_conversion import convert_dict_to_numpy
import numpy as np
import tqdm
import pickle
import evalpy


def step(env, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    """Take an action and return the response of the env."""
    action_dict = {"player_0": action}
    next_state, reward, done, _ = env.step(action_dict)
    next_state = convert_dict_to_numpy(next_state["player_0"])
    done = done["player_0"]
    reward = reward["player_0"]
    return next_state, reward, done


def train(dqn_agent, env, num_steps: int, agent_string, agent_dir, checkpoint_save=50000, evalpy_config=None):
    """Train the agent."""
    dqn_agent.is_test = False

    evalpy.set_project(evalpy_config["project_path"], evalpy_config["project_folder"])

    with evalpy.start_run(evalpy_config["experiment_name"]):

        state = env.reset()["player_0"]
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        checkpoint_counter = 0

        progress_bar = tqdm.tqdm(range(1, num_steps + 1))

        stat = {"Last episode score": 0, "Best Score": 0, "Epsilon": dqn_agent.epsilon, "Loss": 0,
                "Last Episode Length": 0}
        episode_start_frame = 1
        for training_idx in progress_bar:

            action = dqn_agent.select_action(state)
            next_state, reward, done = step(env, action)

            transition = [state, action, reward, next_state, done]
            if not dqn_agent.is_test:
                dqn_agent.store_transition(transition)

            state = next_state
            score += reward

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(training_idx / num_steps, 1.0)
            dqn_agent.beta = dqn_agent.beta + fraction * (1.0 - dqn_agent.beta)

            # if episode ends
            if done:
                episode_end_frame = training_idx
                stat["Last Episode Length"] = episode_end_frame - episode_start_frame + 1
                episode_start_frame = episode_end_frame + 1
                state = env.reset()["player_0"]
                scores.append(score)
                stat["Last episode score"] = float(score)
                stat["Best Score"] = float(max(scores))
                score = 0

                stat["Epsilon"] = dqn_agent.epsilon

                evalpy.log_run_step(stat, step_forward=True)

            # if training is ready
            if len(dqn_agent.memory) >= dqn_agent.init_mem_requirement:
                loss = dqn_agent.update_model()
                losses.append(loss)
                update_cnt += 1
                stat["Loss"] = loss

                # if hard update is needed
                if update_cnt % dqn_agent.target_update == 0:
                    dqn_agent._target_hard_update()

            if training_idx % checkpoint_save == 0:
                with open(f"{agent_dir}checkpoint_{checkpoint_counter}_{agent_string}", "wb") as output_file:
                    pickle.dump(dqn_agent, output_file)
                checkpoint_counter += 1

            progress_bar.set_postfix(stat)
            # plotting
            # if frame_idx % plotting_interval == 0:
            #     plot(frame_idx, scores, losses)

        env.close()

        evalpy.log_run_entries(stat)
        evalpy.log_run_entries(dqn_agent.params())
