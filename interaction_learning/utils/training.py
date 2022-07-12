import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from clean_rl.rl_core.episode import collect_experience
from clean_rl.rl_core.postprocessing import postprocess
from clean_rl.rl_core.ppo_loss import ppo_surrogate_loss
from datetime import datetime
import evalpy
import tqdm


def training_loop(agents, optimizer, agent_configs, env, buffer, training_steps, training_iterations, num_epochs,
                  num_batches, project_path, project_folder, experiment_name, log_dict):
    evalpy.set_project(project_path, project_folder)

    with evalpy.start_run(experiment_name):
        last_episode_stats = {}
        progress_bar = tqdm.tqdm(range(training_iterations))
        for training_iteration in progress_bar:
            episode_stats = collect_experience(env, buffer, agents, training_steps)
            postprocess(agents, buffer, agent_configs)

            for epoch in range(num_epochs):
                for player in agents:
                    for batch in buffer.buffer[player].build_batches(num_batches):
                        loss, stats = ppo_surrogate_loss(agents[player], batch, agent_configs[player])

                        optimizer[player].zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                        optimizer[player].step()

            buffer.reset()
            last_episode_stats = episode_stats

            for idx in range(len(episode_stats["player_0"]["episode_rewards"])):
                stat = {}
                log_stat = {}
                for player in agents:
                    stat[f"{player}_reward"] = episode_stats[player]["episode_rewards"][idx]
                    stat[f"{player}_sampled_steps"] = episode_stats[player]["sampled_steps"][idx]
                    stat[f"training_iteration"] = training_iteration + 1
                    for entry in agent_configs[player]:
                        log_stat[f"{player}_{entry}"] = agent_configs[player][entry]
                log_stat.update(stat)
                progress_bar.set_postfix(stat)
                evalpy.log_run_step(log_stat, step_forward=True)

        agent_stats = {}
        for player_name in agents:
            agent_stats[f"{player_name}_final_reward"] = last_episode_stats[player_name]["reward_mean"]
        agent_stats["timestamp"] = str(datetime.now())
        evalpy.log_run_entries(log_dict)
        evalpy.log_run_entries(agent_stats)

