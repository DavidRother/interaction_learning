import torch


def sac_gradient_step(batch, q_model_1, q_model_2, policy, temperature):
    # batch is states, actions, rewards, next_states, dones
    states = batch[0]
    q_1_loss = q_model_1(torch.tensor(states))


