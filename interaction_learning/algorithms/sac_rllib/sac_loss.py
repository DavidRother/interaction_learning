from interaction_learning.algorithms.sac_rllib.buffer import *
import torch


def actor_critic_loss(policy, model, dist_class, train_batch):
    """Constructs the loss for the Soft Actor Critic.
    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(Buffer(obs=train_batch[CUR_OBS], _is_training=True), [], None)

    model_out_tp1, _ = model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Get all action probs directly from pi and form their logp.
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
    policy_t = torch.exp(log_pis_t)
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
    log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
    policy_tp1 = torch.exp(log_pis_tp1)
    # Q-values.
    q_t, _ = model.get_q_values(model_out_t)
    # Target Q-values.
    q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
    if policy.config["twin_q"]:
        twin_q_t, _ = model.get_twin_q_values(model_out_t)
        twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)
    q_tp1 -= alpha * log_pis_tp1

    # Actually selected Q-values (from the actions batch).
    one_hot = F.one_hot(
        train_batch[ACTIONS].long(), num_classes=q_t.size()[-1]
    )
    q_t_selected = torch.sum(q_t * one_hot, dim=-1)
    if policy.config["twin_q"]:
        twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
    # Discrete case: "Best" means weighted by the policy (prob) outputs.
    q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
    q_tp1_best_masked = (1.0 - train_batch[DONES].float()) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (train_batch[REWARDS] + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    weighted_log_alpha_loss = policy_t.detach() * (
        -model.log_alpha * (log_pis_t + model.target_entropy).detach()
    )
    # Sum up weighted terms and mean over all batch items.
    alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
    # Actor loss.
    actor_loss = torch.mean(
        torch.sum(
            torch.mul(
                # NOTE: No stop_grad around policy output here
                # (compare with q_t_det_policy for continuous case).
                policy_t,
                alpha.detach() * log_pis_t - q_t.detach(),
            ),
            dim=-1,
        )
        )


    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])
