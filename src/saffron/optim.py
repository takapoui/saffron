from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class ConstantScheduleConfig:
    pass


@dataclass
class CosineScheduleConfig:
    warmup_steps: int
    min_lr_ratio: float


ScheduleConfig = ConstantScheduleConfig | CosineScheduleConfig


def _schedule_from_dict(d: dict[str, Any]) -> ScheduleConfig:
    schedule_type = d.get("type", "constant")
    if schedule_type == "cosine":
        return CosineScheduleConfig(
            warmup_steps=d["warmup_steps"],
            min_lr_ratio=d["min_lr_ratio"],
        )
    if schedule_type == "constant":
        return ConstantScheduleConfig()
    raise ValueError(f"Unknown schedule type: {schedule_type!r}")


@dataclass
class OptimizerConfig:
    optimizer_type: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    schedule: ScheduleConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OptimizerConfig:
        betas_raw = d["betas"]
        schedule = _schedule_from_dict(d["schedule"])
        return cls(
            optimizer_type=d["type"],
            lr=d["lr"],
            weight_decay=d["weight_decay"],
            betas=(betas_raw[0], betas_raw[1]),
            eps=d["eps"],
            schedule=schedule,
        )


def configure_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    device_type: str,
) -> torch.optim.Optimizer:
    if config.optimizer_type == "adamw":
        return _configure_adamw(model, config, device_type)
    raise ValueError(f"Unknown optimizer type: {config.optimizer_type!r}")


def _configure_adamw(
    model: nn.Module,
    config: OptimizerConfig,
    device_type: str,
) -> torch.optim.AdamW:
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay_params = [param for param in param_dict.values() if param.ndim >= 2]
    no_decay_params = [param for param in param_dict.values() if param.ndim <= 1]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    logger.info(
        f"decayed parameters: {len(decay_params)} groups, "
        f"{sum(param.numel() for param in decay_params)}"
    )
    logger.info(
        f"nondecayed parameters: {len(no_decay_params)} groups, "
        f"{sum(param.numel() for param in no_decay_params)}"
    )
    use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    logger.info(f"using fused AdamW: {use_fused}")
    return torch.optim.AdamW(
        optim_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        fused=use_fused,
    )


def get_lr(step: int, max_steps: int, config: OptimizerConfig) -> float:
    s = config.schedule
    if isinstance(s, ConstantScheduleConfig):
        return config.lr
    else:
        assert isinstance(s, CosineScheduleConfig), f"Unknown schedule type: {type(s).__name__!r}"
        min_lr = config.lr * s.min_lr_ratio
        if s.warmup_steps > 0 and step < s.warmup_steps:
            return config.lr * (step + 1) / s.warmup_steps
        if step >= max_steps or max_steps <= s.warmup_steps:
            return min_lr
        decay_ratio = (step - s.warmup_steps) / (max_steps - s.warmup_steps)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (config.lr - min_lr)
