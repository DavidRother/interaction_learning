import tqdm
import pickle
import evalpy


def gather_actions(agents, state):
    return {agent: agents[agent].select_action(state[agent]) for agent in agents}


def train(agents, env, num_steps: int, agent_string, agent_dir, checkpoint_save=50000, evalpy_config=None):
    """Train the agent."""

    evalpy.set_project(evalpy_config["project_path"], evalpy_config["project_folder"])

    with evalpy.start_run(evalpy_config["experiment_name"]):

        state = env.reset()
        update_cnt = 0
        losses = {agent: [] for agent in agents}
        scores = {agent: [] for agent in agents}
        score = {agent: 0 for agent in agents}
        checkpoint_ctr = 0

        progress_bar = tqdm.tqdm(range(1, num_steps + 1))
        stat = {}
        for agent in agents:
            stat.update({f"Last episode score {agent}": 0, f"Best Score {agent}": 0})
            try:
                stat.update(agents[agent].stats(agent))
            except AttributeError:
                pass
        stat.update({"Last Episode Length": 0})
        episode_start_frame = 1
        for training_idx in progress_bar:

            actions = gather_actions(agents, state)
            next_state, reward, done, _ = env.step(actions)

            for agent in agents:
                if agents[agent].is_test:
                    pass
                transition = [state[agent], actions[agent], reward[agent], next_state[agent], done[agent]]
                agents[agent].store_transition(transition)

            state = next_state
            for agent in agents:
                score[agent] += reward[agent]

            for agent in agents:
                try:
                    fraction = min(training_idx / num_steps, 1.0)
                    agents[agent].postprocess_step(fraction)
                except AttributeError:
                    pass

            # PER: increase beta
            # fraction = min(training_idx / num_steps, 1.0)
            # dqn_agent.beta = dqn_agent.beta + fraction * (1.0 - dqn_agent.beta)

            # if episode ends
            if all(done.values()):
                episode_end_frame = training_idx
                stat["Last Episode Length"] = episode_end_frame - episode_start_frame + 1
                episode_start_frame = episode_end_frame + 1
                state = env.reset()
                for agent in agents:
                    scores[agent].append(score[agent])
                    stat[f"Last episode score {agent}"] = float(score[agent])
                    stat[f"Best Score {agent}"] = float(max(scores[agent]))
                    score[agent] = 0

                    try:
                        stat.update(agents[agent].stats(agent))
                    except AttributeError:
                        pass

                evalpy.log_run_step(stat, step_forward=True)

            # if training is ready
            for agent in agents:
                if agents[agent].is_test:
                    continue
                try:
                    if agents[agent].training_required:
                        loss = agents[agent].update_model()
                        losses[agent].append(loss)
                        update_cnt += 1
                        for loss_descriptor in loss:
                            stat[f"{loss_descriptor} {agent}"] = loss[loss_descriptor]

                        # if hard update is needed
                        if update_cnt % agents[agent].target_update == 0:
                            agents[agent].target_hard_update()
                except AttributeError:
                    continue

            if training_idx % checkpoint_save == 0:
                for agent in agents:
                    with open(f"{agent_dir}checkpoint_{checkpoint_ctr}_{agent}_{agent_string}", "wb") as output_file:
                        pickle.dump(agents[agent], output_file)
                checkpoint_ctr += 1

            progress_bar.set_postfix(stat)
            # plotting
            # if frame_idx % plotting_interval == 0:
            #     plot(frame_idx, scores, losses)

        env.close()

        evalpy.log_run_entries(stat)
        for agent in agents:
            evalpy.log_run_entries(agents[agent].params(agent))
