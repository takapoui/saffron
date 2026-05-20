import torch


def compute_grpo_loss(
    new_log_probs: torch.Tensor,  # (B, T-1) — current policy, requires_grad
    old_log_probs: torch.Tensor,  # (B, T-1) — cached at rollout time, no grad
    ref_log_probs: torch.Tensor,  # (B, T-1) — frozen reference model, no grad
    advantages: torch.Tensor,  # (B, T-1) — broadcast per-completion scalar
    response_mask: torch.Tensor,  # (B, T-1) — 1 on completion tokens, 0 elsewhere
    clip_eps: float,  # ε in min(r·A, clip(r, 1±ε)·A); paper uses 0.2
    kl_coef: float,  # β on the KL penalty; paper uses ~1e-3
    total_response_len: torch.Tensor,  # scalar — full-batch token count (shared across mbs)
) -> tuple[torch.Tensor, dict[str, float]]:  # (scalar_loss, metrics)

    kl_log_diff = ref_log_probs - new_log_probs
    kl_penalty = (torch.exp(kl_log_diff) - 1 - kl_log_diff) * response_mask

    policy_ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate_1 = policy_ratio * advantages
    surrogate_2 = torch.clamp(policy_ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    policy_loss = -torch.minimum(surrogate_1, surrogate_2) * response_mask

    # Per-token loss: every response token weighted equally across the full batch.
    # total_response_len is computed once over the entire batch and passed in, so each
    # microbatch contributes its sum to a shared denominator. Backward() across microbatches
    # then accumulates exactly the full-batch gradient.
    loss = (policy_loss + kl_coef * kl_penalty).sum() / total_response_len

    with torch.no_grad():
        # All per-token metrics divide by the same full-batch total so they aggregate
        # cleanly: summing across microbatches gives the correct full-batch values.
        clip_active = (surrogate_2 < surrogate_1).float() * response_mask
        metrics = {
            "policy_loss": policy_loss.sum().item() / total_response_len.item(),
            "kl": kl_penalty.sum().item() / total_response_len.item(),
            "approximate_entropy": (
                (-new_log_probs * response_mask).sum().item() / total_response_len.item()
            ),
            "clip_fraction": clip_active.sum().item() / total_response_len.item(),
        }
    return loss, metrics
