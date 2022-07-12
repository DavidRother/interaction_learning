from torch.distributions.categorical import Categorical
import torch


def collect_experience(env, buffer, agents, steps):
    done_steps = 0
    stats = stats_init(agents)
    while True:
        obs = env.reset()
        t = 0
        done = {player: False for player in agents}
        reward_acc = {player: 0 for player in agents}
        while not any(done.values()):
            actions = {}
            logits = {}
            logits_numpy = {}
            for idx, agent in enumerate(agents):
                agent_obs = torch.unsqueeze(torch.Tensor(obs[f"player_{idx}"]), 0)
                logits[f"player_{idx}"] = agents[agent](agent_obs).squeeze(1)
                logits_numpy[f"player_{idx}"] = agents[agent](agent_obs).squeeze(1).detach().cpu().numpy()
                actions[f"player_{idx}"] = Categorical(logits=logits[f"player_{idx}"]).sample().item()
            next_obs, rewards, done, info = env.step(actions)
            for player in agents:
                reward_acc[player] += rewards[player]

            buffer.commit(obs, actions, next_obs, rewards, done, info, logits_numpy, t)

            obs = next_obs
            done_steps += 1
            t += 1
            if done_steps >= steps:
                break

        for player in agents:
            stats[player]["total_episodes"] += 1
            stats[player]["reward_max"] = max(stats[player]["reward_max"], reward_acc[player])
            stats[player]["reward_min"] = min(stats[player]["reward_min"], reward_acc[player])
            stats[player]["reward_mean"] = (stats[player]["reward_mean"] * (stats[player]["total_episodes"] - 1) +
                                            reward_acc[player]) / stats[player]["total_episodes"]
            stats[player]["episode_rewards"].append(reward_acc[player])
            stats[player]["sampled_steps"].append(steps)

        if done_steps >= steps:
            break
    if done_steps != steps:
        raise Exception("How")
    return stats


def stats_init(agents):
    stats = {}
    for agent in agents:
        stats[agent] = {"reward_mean": 0, "reward_min": 0, "reward_max": 0, "total_episodes": 0, "episode_rewards": [],
                        "sampled_steps": []}
    return stats
