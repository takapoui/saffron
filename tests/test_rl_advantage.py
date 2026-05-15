"""Tests for compute_grpo_advantages (GRPO group normalization)."""

from __future__ import annotations

import pytest

from saffron.rl.advantage import compute_grpo_advantages


def test_single_group_normalized() -> None:
    """One group of 3 rewards: advantages should sum to ~0 and use std normalization."""
    rewards = [0.0, 1.0, 2.0]  # mean=1.0, std=sqrt(2/3) ≈ 0.816
    lens = [4, 4, 4]
    out = compute_grpo_advantages(rewards, response_lens=lens, group_size=3)

    # Output shape matches response_lens
    assert [len(row) for row in out] == lens

    # Each row is constant (broadcast)
    for row in out:
        assert len(set(row)) == 1

    # Within-group advantages should center around 0
    scalars = [row[0] for row in out]
    assert sum(scalars) == pytest.approx(0.0, abs=1e-5)  # type: ignore[reportUnknownMemberType]

    # Order preserved: lowest reward → most negative advantage
    assert scalars[0] < scalars[1] < scalars[2]


def test_all_rewards_equal_gives_zero_advantages() -> None:
    """When std=0, normalization with eps gives advantages near 0 (no NaN)."""
    rewards = [0.5, 0.5, 0.5, 0.5]
    lens = [3, 3, 3, 3]
    out = compute_grpo_advantages(rewards, response_lens=lens, group_size=4)

    for row in out:
        for val in row:
            assert val == pytest.approx(0.0, abs=1e-3)  # type: ignore[reportUnknownMemberType]


def test_multiple_groups_normalized_independently() -> None:
    """Two groups: each normalized within itself, not across groups."""
    # Group 1: rewards small range
    # Group 2: rewards large range with same mean
    rewards = [0.0, 1.0, 2.0, 10.0, 20.0, 30.0]
    lens = [1, 1, 1, 1, 1, 1]
    out = compute_grpo_advantages(rewards, response_lens=lens, group_size=3)

    scalars = [row[0] for row in out]
    # Each group should center around 0 within itself
    assert sum(scalars[:3]) == pytest.approx(0.0, abs=1e-5)  # type: ignore[reportUnknownMemberType]
    assert sum(scalars[3:]) == pytest.approx(0.0, abs=1e-5)  # type: ignore[reportUnknownMemberType]

    # The 30.0 (top of group 2) and the 2.0 (top of group 1) should both be
    # the *most positive* in their group — same sign, similar magnitude (~1.22),
    # because group normalization is scale-invariant.
    assert scalars[2] == pytest.approx(scalars[5], abs=1e-3)  # type: ignore[reportUnknownMemberType]


def test_broadcasting_matches_response_lens() -> None:
    """Each completion's per-token list has length equal to its response_len."""
    rewards = [1.0, 0.0]
    lens = [5, 10]
    out = compute_grpo_advantages(rewards, response_lens=lens, group_size=2)

    assert len(out) == 2
    assert len(out[0]) == 5
    assert len(out[1]) == 10


def test_broadcasting_value_constant_within_completion() -> None:
    """All tokens of a single completion get the same advantage value."""
    rewards = [0.0, 1.0]
    lens = [3, 4]
    out = compute_grpo_advantages(rewards, response_lens=lens, group_size=2)

    assert out[0][0] == out[0][1] == out[0][2]
    assert out[1][0] == out[1][1] == out[1][2] == out[1][3]


def test_assertion_on_mismatched_lengths() -> None:
    """rewards and response_lens must have the same length."""
    with pytest.raises(AssertionError):
        compute_grpo_advantages(rewards=[1.0, 2.0], response_lens=[3], group_size=2)


def test_assertion_on_non_multiple_group_size() -> None:
    """len(rewards) must be divisible by group_size."""
    with pytest.raises(AssertionError):
        compute_grpo_advantages(rewards=[1.0, 2.0, 3.0], response_lens=[1, 1, 1], group_size=2)
