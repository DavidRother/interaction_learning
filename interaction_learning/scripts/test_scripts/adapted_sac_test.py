import torch
from itertools import count
import gym
import os
import yaml  # pip install pyyaml
from datetime import datetime
from interaction_learning.algorithms.sacd.agent.soft_actor_critic_agent import SACAgent


device = torch.device("cpu")

config_path = os.path.join('../../algorithms/sacd/config', 'sacd.yaml')
env_id = 'CartPole-v0'
shared = False
cuda = False
seed = 0

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Specify the directory to log.
name = config_path.split('/')[-1].rstrip('.yaml')
if shared:
    name = 'shared-' + name
time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join(
    'logs', env_id, f'{name}-seed{seed}-{time}')

# Create environments.
env = gym.make(env_id)
test_env = gym.make(env_id)

obs_space = env.observation_space
action_space = env.action_space

agent = SACAgent(obs_space, action_space, log_dir=log_dir, cuda=cuda,
                 seed=seed, **config)

learn_steps = 0
begin_learn = False
episode_reward = 0
agent.steps = 0

for epoch in count():
    state = env.reset()
    episode_reward = 0
    for time_steps in range(200):
        action = agent.select_action(state, explore=True)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.memory.append(state, action, reward, next_state, done)
        agent.steps += 1
        if agent.is_update():
            if begin_learn is False:
                print('begin learning!')
                begin_learn = True
            loss = agent.update_model()
            learn_steps += 1
            if learn_steps % agent.target_update_interval == 0:
                agent.target_hard_update()

        if done:
            break

        state = next_state
    if epoch % 10 == 0:
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

    if epoch % (agent.eval_interval / 200) == 0:
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= 200:
                action = agent.select_action(state, explore=False)
                next_state, reward, done, _ = test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > agent.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        print(f"Evaluation Mean Score: {mean_return}")

    if epoch >= agent.num_steps // 20:
        agent.save_models(os.path.join(agent.model_dir, 'final'))
        break
