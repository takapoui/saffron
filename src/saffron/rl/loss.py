import torch


def compute_grpo_loss(
    new_log_probs: torch.Tensor,  # (B, T-1) — current policy, requires_grad
    old_log_probs: torch.Tensor,  # (B, T-1) — cached at rollout time, no grad
    ref_log_probs: torch.Tensor,  # (B, T-1) — frozen reference model, no grad
    advantages: torch.Tensor,  # (B, T-1) — broadcast per-completion scalar
    response_mask: torch.Tensor,  # (B, T-1) — 1 on completion tokens, 0 elsewhere
    clip_eps: float,  # ε in min(r·A, clip(r, 1±ε)·A); paper uses 0.2
    kl_coef: float,  # β on the KL penalty; paper uses ~1e-3
) -> tuple[torch.Tensor, dict[str, float]]:  # (scalar_loss, metrics)

    kl_log_diff = ref_log_probs - new_log_probs
    kl_penalty = (torch.exp(kl_log_diff) - 1 - kl_log_diff) * response_mask

    policy_ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate_1 = policy_ratio * advantages
    surrogate_2 = torch.clamp(policy_ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    policy_loss = -torch.minimum(surrogate_1, surrogate_2) * response_mask

    denom = response_mask.sum(-1).clamp(min=1)
    seq_loss = (policy_loss + kl_coef * kl_penalty).sum(-1) / denom
    loss = seq_loss.mean()

    with torch.no_grad():
        per_completion_adv = (advantages * response_mask).sum(-1) / denom
        metrics = {
            "policy_loss": (policy_loss.sum(-1) / denom).mean().item(),
            "kl": (kl_penalty.sum(-1) / denom).mean().item(),
            "zero_advantage_ratio": (per_completion_adv.abs() < 1e-6).float().mean().item(),
            "approximate_entropy": (
                (-new_log_probs * response_mask).sum() / response_mask.sum().clamp(min=1)
            ).item(),
        }
    return loss, metrics
