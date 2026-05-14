"""Tests for learning-rate schedule and AdamW parameter groups."""

from __future__ import annotations

import pytest
import torch

from saffron.config import OptimizerConfig, ScheduleConfig
from saffron.model import GPT2, GPT2Config
from saffron.optim import configure_adamw, get_lr_cosine

# ---------------------------------------------------------------------------
# LR schedule constants
# ---------------------------------------------------------------------------

MAX_LR = 1e-3
MIN_LR_RATIO = 0.1
MIN_LR = MAX_LR * MIN_LR_RATIO
WARMUP = 10
MAX_STEPS = 100
SCHEDULE = ScheduleConfig(warmup_steps=WARMUP, min_lr_ratio=MIN_LR_RATIO)


# ---------------------------------------------------------------------------
# LR schedule tests
# ---------------------------------------------------------------------------


def test_warmup_start() -> None:
    lr = get_lr_cosine(step=0, max_steps=MAX_STEPS, max_lr=MAX_LR, config=SCHEDULE)
    assert lr == pytest.approx(MAX_LR / WARMUP)  # type: ignore[reportUnknownMemberType]


def test_warmup_peak() -> None:
    lr = get_lr_cosine(
        step=WARMUP - 1,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        config=SCHEDULE,
    )
    assert lr == pytest.approx(MAX_LR)  # type: ignore[reportUnknownMemberType]


def test_post_warmup_below_peak() -> None:
    lr = get_lr_cosine(
        step=WARMUP + 1,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        config=SCHEDULE,
    )
    assert lr < MAX_LR


def test_end_is_min_lr() -> None:
    # at step == max_steps the clamp kicks in and returns exactly min_lr
    lr = get_lr_cosine(
        step=MAX_STEPS,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        config=SCHEDULE,
    )
    assert lr == pytest.approx(MIN_LR)  # type: ignore[reportUnknownMemberType]


def test_beyond_max_steps_clamps() -> None:
    lr = get_lr_cosine(
        step=MAX_STEPS + 50,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        config=SCHEDULE,
    )
    assert lr == pytest.approx(MIN_LR)  # type: ignore[reportUnknownMemberType]


def test_monotone_decay_after_warmup() -> None:
    lrs = [
        get_lr_cosine(step=s, max_steps=MAX_STEPS, max_lr=MAX_LR, config=SCHEDULE)
        for s in range(WARMUP, MAX_STEPS)
    ]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ---------------------------------------------------------------------------
# AdamW parameter-group fixtures and tests
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> GPT2Config:
    return GPT2Config(
        vocab_size=1000,
        n_embd=96,
        block_size=64,
        n_layer=2,
        n_head=4,
    )


@pytest.fixture
def model(config: GPT2Config) -> GPT2:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return GPT2(config)


def test_adamw_2d_params_get_weight_decay(model: GPT2) -> None:
    optimizer = configure_adamw(
        model, config=OptimizerConfig(lr=1e-3, weight_decay=0.1), device_type="cpu"
    )
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.ndim >= 2:
                assert group["weight_decay"] == 0.1


def test_adamw_1d_params_no_weight_decay(model: GPT2) -> None:
    optimizer = configure_adamw(
        model, config=OptimizerConfig(lr=1e-3, weight_decay=0.1), device_type="cpu"
    )
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.ndim <= 1:
                assert group["weight_decay"] == 0.0
