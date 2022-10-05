

def evaluate(env, num_episodes, agents):
    episode_rewards = []
    for epoch in range(num_episodes):
        state = env.reset()
        episode_reward = {f"player_{idx}": 0 for idx in range(len(agents))}
        done = {f"player_{idx}": False for idx in range(len(agents))}
        while not all(done.values()):
            actions = {f"player_{idx}": agent.select_action(state[f"player_{idx}"]) for idx, agent in enumerate(agents)}
            next_state, reward, done, _ = env.step(actions)
            for agent, rew in reward.items():
                episode_reward[agent] += rew

            state = next_state

        episode_rewards.append(episode_reward)

    return episode_rewards


class DummyAgent:

    def __init__(self):
        pass

    def select_action(self, state):
        return 0

