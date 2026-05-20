"""Tests for learning-rate schedule and AdamW parameter groups."""

from __future__ import annotations

import pytest
import torch

from saffron.model import GPT2, GPT2Config
from saffron.optim import (
    ConstantScheduleConfig,
    CosineScheduleConfig,
    OptimizerConfig,
    configure_optimizer,
    get_lr,
)

# ---------------------------------------------------------------------------
# LR schedule constants
# ---------------------------------------------------------------------------

MAX_LR = 1e-3
MIN_LR_RATIO = 0.1
MIN_LR = MAX_LR * MIN_LR_RATIO
WARMUP = 10
MAX_STEPS = 100
SCHEDULE = CosineScheduleConfig(warmup_steps=WARMUP, min_lr_ratio=MIN_LR_RATIO)
OPTIM_CFG = OptimizerConfig(
    optimizer_type="adamw",
    lr=MAX_LR,
    weight_decay=0.0,
    betas=(0.9, 0.95),
    eps=1e-8,
    schedule=SCHEDULE,
)


# ---------------------------------------------------------------------------
# LR schedule tests
# ---------------------------------------------------------------------------


def test_warmup_start() -> None:
    lr = get_lr(step=0, max_steps=MAX_STEPS, config=OPTIM_CFG)
    assert lr == pytest.approx(MAX_LR / WARMUP)  # type: ignore[reportUnknownMemberType]


def test_warmup_peak() -> None:
    lr = get_lr(step=WARMUP - 1, max_steps=MAX_STEPS, config=OPTIM_CFG)
    assert lr == pytest.approx(MAX_LR)  # type: ignore[reportUnknownMemberType]


def test_post_warmup_below_peak() -> None:
    lr = get_lr(step=WARMUP + 1, max_steps=MAX_STEPS, config=OPTIM_CFG)
    assert lr < MAX_LR


def test_end_is_min_lr() -> None:
    lr = get_lr(step=MAX_STEPS, max_steps=MAX_STEPS, config=OPTIM_CFG)
    assert lr == pytest.approx(MIN_LR)  # type: ignore[reportUnknownMemberType]


def test_beyond_max_steps_clamps() -> None:
    lr = get_lr(step=MAX_STEPS + 50, max_steps=MAX_STEPS, config=OPTIM_CFG)
    assert lr == pytest.approx(MIN_LR)  # type: ignore[reportUnknownMemberType]


def test_monotone_decay_after_warmup() -> None:
    lrs = [get_lr(step=s, max_steps=MAX_STEPS, config=OPTIM_CFG) for s in range(WARMUP, MAX_STEPS)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_constant_schedule_returns_lr() -> None:
    cfg = OptimizerConfig(
        optimizer_type="adamw",
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        schedule=ConstantScheduleConfig(),
    )
    for step in [0, 10, 100]:
        assert get_lr(step=step, max_steps=MAX_STEPS, config=cfg) == pytest.approx(1e-3)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# AdamW parameter-group fixtures and tests
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=1000,
        n_embd=96,
        block_size=64,
        n_layer=2,
        n_head=4,
    )


@pytest.fixture
def model(gpt2_config: GPT2Config) -> GPT2:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return GPT2(gpt2_config)


def _adamw_cfg(weight_decay: float) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer_type="adamw",
        lr=1e-3,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        schedule=ConstantScheduleConfig(),
    )


def test_2d_params_get_weight_decay(model: GPT2) -> None:
    optimizer = configure_optimizer(model, config=_adamw_cfg(0.1), device_type="cpu")
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.ndim >= 2:
                assert group["weight_decay"] == 0.1


def test_1d_params_no_weight_decay(model: GPT2) -> None:
    optimizer = configure_optimizer(model, config=_adamw_cfg(0.1), device_type="cpu")
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.ndim <= 1:
                assert group["weight_decay"] == 0.0
