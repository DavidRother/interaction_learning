from time import sleep


def evaluate(env, num_episodes, agents, kwargs_agents=None, render=False):
    kwargs_agents = kwargs_agents or [{}] * len(agents)
    episode_rewards = []
    for epoch in range(num_episodes):
        state = env.reset()

        if render:
            env.render()
            sleep(0.1)

        episode_reward = {f"player_{idx}": 0 for idx in range(len(agents))}
        done = {f"player_{idx}": False for idx in range(len(agents))}
        while not all(done.values()):
            actions = {f"player_{idx}": agent.select_action(state[f"player_{idx}"], **kwargs) for idx, (agent, kwargs) in enumerate(zip(agents, kwargs_agents))}
            next_state, reward, done, truncation, _ = env.step(actions)
            for agent, rew in reward.items():
                episode_reward[agent] += rew

            state = next_state

            if render:
                env.render()
                sleep(0.1)

        episode_rewards.append(episode_reward)

    return episode_rewards


class DummyAgent:

    def __init__(self):
        pass

    def select_action(self, state):
        return 0

