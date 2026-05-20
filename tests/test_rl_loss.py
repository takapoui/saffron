"""Tests for compute_grpo_loss (GRPO clipped surrogate + KL penalty)."""

from __future__ import annotations

import pytest
import torch

from saffron.rl.loss import compute_grpo_loss


def _ones_mask(B: int, T_minus_1: int) -> torch.Tensor:
    return torch.ones((B, T_minus_1), dtype=torch.float32)


def _total_len(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum().clamp(min=1)


def test_returns_scalar_and_metrics_dict() -> None:
    """Output is (scalar tensor, dict of floats) with the expected keys."""
    B, T = 2, 4
    new_lp = torch.zeros((B, T - 1), requires_grad=True)
    old_lp = torch.zeros((B, T - 1))
    ref_lp = torch.zeros((B, T - 1))
    adv = torch.ones((B, T - 1))
    mask = _ones_mask(B, T - 1)

    loss, metrics = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.001,
        total_response_len=_total_len(mask),
    )

    assert loss.ndim == 0  # scalar
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) >= {
        "policy_loss",
        "kl",
        "approximate_entropy",
        "clip_fraction",
    }
    for v in metrics.values():
        assert isinstance(v, float)


def test_ratio_one_reduces_to_negative_advantage() -> None:
    """When new == old, ratio=1, clipping is a no-op. Loss equals -advantage (KL=0 if ref==new)."""
    B, T = 2, 4
    new_lp = torch.full((B, T - 1), -1.5, requires_grad=True)
    old_lp = new_lp.detach().clone()
    ref_lp = new_lp.detach().clone()  # KL term is exactly 0
    adv_value = 0.7
    adv = torch.full((B, T - 1), adv_value)
    mask = _ones_mask(B, T - 1)

    loss, metrics = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.1,
        total_response_len=_total_len(mask),
    )

    # loss = -ratio * adv = -1 * 0.7 = -0.7 (sum / total_len cancels)
    assert loss.item() == pytest.approx(-adv_value, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_kl_is_zero_when_ref_equals_new() -> None:
    """k3 KL estimator is exactly 0 when log-probs match: exp(0) - 0 - 1 = 0."""
    B, T = 1, 3
    lp = torch.full((B, T - 1), -0.5)
    mask = _ones_mask(B, T - 1)
    loss, metrics = compute_grpo_loss(
        new_log_probs=lp.clone().requires_grad_(True),
        old_log_probs=lp.clone(),
        ref_log_probs=lp.clone(),
        advantages=torch.zeros((B, T - 1)),
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=1.0,
        total_response_len=_total_len(mask),
    )
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-7)  # type: ignore[reportUnknownMemberType]
    # No advantage signal → policy_loss should also be 0
    assert loss.item() == pytest.approx(0.0, abs=1e-7)  # type: ignore[reportUnknownMemberType]


def test_clipping_activates_with_positive_advantage_and_large_ratio() -> None:
    """When ratio > 1+ε and advantage > 0, surrogate_1 > surrogate_2 → clipped term wins minimum."""
    B, T = 1, 2
    # Force ratio = exp(0.5) ≈ 1.648, which exceeds 1 + 0.2 = 1.2
    new_lp = torch.full((B, T - 1), 0.5, requires_grad=True)
    old_lp = torch.zeros((B, T - 1))
    ref_lp = torch.zeros((B, T - 1))  # we don't care about KL here
    adv = torch.ones((B, T - 1))  # positive
    mask = _ones_mask(B, T - 1)

    loss, _ = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.0,  # isolate the surrogate
        total_response_len=_total_len(mask),
    )

    # With clip: surrogate_2 = 1.2 * 1.0 = 1.2 → loss = -1.2
    # Without clip would be: -exp(0.5) ≈ -1.648
    # The pessimistic minimum gives the *smaller* objective = larger loss, so loss = -1.2
    assert loss.item() == pytest.approx(-1.2, abs=1e-5)  # type: ignore[reportUnknownMemberType]


def test_clipping_does_not_activate_within_band() -> None:
    """Inside [1-ε, 1+ε], clipped == unclipped, loss matches unclipped surrogate."""
    B, T = 1, 2
    new_lp = torch.full((B, T - 1), 0.1, requires_grad=True)  # ratio = exp(0.1) ≈ 1.105
    old_lp = torch.zeros((B, T - 1))
    ref_lp = torch.zeros((B, T - 1))
    adv = torch.ones((B, T - 1))
    mask = _ones_mask(B, T - 1)

    loss, _ = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.0,
        total_response_len=_total_len(mask),
    )

    # No clipping: loss = -ratio * adv = -exp(0.1)
    assert loss.item() == pytest.approx(-torch.exp(torch.tensor(0.1)).item(), abs=1e-5)  # type: ignore[reportUnknownMemberType]


def test_mask_zero_positions_do_not_contribute() -> None:
    """Padded positions (mask=0) must not affect the loss value."""
    B, T = 1, 4
    new_lp = torch.full((B, T - 1), -1.0, requires_grad=True)
    old_lp = torch.full((B, T - 1), -1.0)
    ref_lp = torch.full((B, T - 1), -1.0)
    adv = torch.tensor([[1.0, 1.0, 999.0]])  # huge value on a position we'll mask out
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss, _ = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.0,
        total_response_len=_total_len(mask),
    )

    # Only first two positions count; loss = sum(-1*1, -1*1) / 2 = -1.0
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)  # type: ignore[reportUnknownMemberType]


def test_gradient_flows_only_through_new_log_probs() -> None:
    """Backward should populate .grad on new_log_probs only — not on old or ref."""
    B, T = 2, 3
    new_lp = torch.zeros((B, T - 1), requires_grad=True)
    old_lp = torch.zeros((B, T - 1), requires_grad=False)
    ref_lp = torch.zeros((B, T - 1), requires_grad=False)
    adv = torch.ones((B, T - 1))
    mask = _ones_mask(B, T - 1)

    loss, _ = compute_grpo_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        ref_log_probs=ref_lp,
        advantages=adv,
        response_mask=mask,
        clip_eps=0.2,
        kl_coef=0.01,
        total_response_len=_total_len(mask),
    )
    loss.backward()  # type: ignore[reportUnknownMemberType]

    assert new_lp.grad is not None
    assert old_lp.grad is None
    assert ref_lp.grad is None
