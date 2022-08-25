from gym_cooking.environment.game.game import Game


def gather_human_examples(agent, env, max_steps, num_trajectories):
    agent_string = "player_0"
    for _ in range(num_trajectories):
        game = Game(env, 1, [], max_steps)
        store = game.on_execute()
        for idx in range(len(store["agent_states"])):
            transition = [store["observation"][idx][agent_string], store["actions"][idx][agent_string],
                          store["rewards"][idx][agent_string], store["observation"][idx + 1][agent_string],
                          store["done"][idx][agent_string]]
            agent.store_transition(transition)


def gather_expert_examples(expert, agent, env, num_trajectories):
    agent_string = "player_0"
    steps_gathered = 0
    for idx in range(num_trajectories):
        state = env.reset()
        done = {"player_0": False}
        while not all(done.values()):
            actions = {agent_string: expert.select_action(state[agent_string])}
            next_state, reward, done, _ = env.step(actions)
            transition = [state[agent_string], actions[agent_string], reward[agent_string], next_state[agent_string],
                          done[agent_string]]
            agent.store_transition(transition)
            steps_gathered += 1
            state = next_state

    print(f"Gathered Expert Steps: {steps_gathered}")
