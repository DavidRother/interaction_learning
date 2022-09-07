import torch
import torch.nn.functional as F
from itertools import count
import gym
from interaction_learning.algorithms.soft_dqn.soft_dqn_utils import Memory, SoftQNetwork


device = torch.device("cpu")

env = gym.make('CartPole-v0')

alpha = 4.0

onlineQNetwork1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, 4.0, device="cpu").to(device)
targetQNetwork1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, 4.0, device="cpu").to(device)
targetQNetwork1.load_state_dict(onlineQNetwork1.state_dict())

onlineQNetwork2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, 4.0, device="cpu").to(device)
targetQNetwork2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.n, 4.0, device="cpu").to(device)
targetQNetwork2.load_state_dict(onlineQNetwork2.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork2.parameters(), lr=1e-4)

GAMMA = 0.99
REPLAY_MEMORY = 50000
BATCH = 16
UPDATE_STEPS = 4

memory_replay = Memory(REPLAY_MEMORY)

learn_steps = 0
begin_learn = False
episode_reward = 0

for epoch in count():
    state = env.reset()
    episode_reward = 0
    for time_steps in range(200):
        action = onlineQNetwork.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        memory_replay.add((state, next_state, action, reward, done))

        if memory_replay.size() > 128:
            if begin_learn is False:
                print('learn begin!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                next_q = targetQNetwork(batch_next_state)
                next_v = targetQNetwork.get_value(next_q)
                y = batch_reward + (1 - batch_done) * GAMMA * next_v

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

        state = next_state
    print(episode_reward)
    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'sql-policy.para')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
