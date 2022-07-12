import torch
from torch.nn.utils import clip_grad_norm_


def update_model(dqn_agent) -> torch.Tensor:
    """Update the model by gradient descent."""
    # PER needs beta to calculate weights
    samples = dqn_agent.memory.sample_batch(dqn_agent.beta)
    weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(dqn_agent.device)
    indices = samples["indices"]

    # 1-step Learning loss
    elementwise_loss = dqn_agent._compute_dqn_loss(samples, dqn_agent.gamma)

    # PER: importance sampling before average
    loss = torch.mean(elementwise_loss * weights)

    # N-step Learning loss
    # we are gonna combine 1-step loss and n-step loss so as to
    # prevent high-variance. The original rainbow employs n-step loss only.
    if dqn_agent.use_n_step:
        gamma = dqn_agent.gamma ** dqn_agent.n_step
        samples = dqn_agent.memory_n.sample_batch_from_idxs(indices)
        elementwise_loss_n_loss = dqn_agent._compute_dqn_loss(samples, gamma)
        elementwise_loss += elementwise_loss_n_loss

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

    dqn_agent.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(dqn_agent.dqn.parameters(), 10.0)
    dqn_agent.optimizer.step()

    # PER: update priorities
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior + dqn_agent.prior_eps
    dqn_agent.memory.update_priorities(indices, new_priorities)

    # NoisyNet: reset noise
    dqn_agent.dqn.reset_noise()
    dqn_agent.dqn_target.reset_noise()

    return loss.item()
