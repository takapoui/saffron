from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval import EvalGenerateConfig, EvalGSM8KConfig, EvalHellaswagConfig, EvalLossConfig
from .model.config import GPT2Config as ModelConfig  # backward compat for old checkpoints

__all__ = ["ModelConfig"]  # ensure unpickling finds it


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OptimizerConfig:
        return cls(lr=d["lr"], weight_decay=d["weight_decay"])


@dataclass
class ScheduleConfig:
    warmup_steps: int
    min_lr_ratio: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScheduleConfig:
        return cls(warmup_steps=d["warmup_steps"], min_lr_ratio=d["min_lr_ratio"])


@dataclass
class TrainConfig:
    # training loop
    max_steps: int
    grad_clip: float
    total_batch_size: int

    # optimizer + schedule
    optimizer: OptimizerConfig
    schedule: ScheduleConfig

    # eval
    eval_loss: EvalLossConfig
    eval_generate: EvalGenerateConfig
    eval_hellaswag: EvalHellaswagConfig
    eval_gsm8k: EvalGSM8KConfig

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None
    resume_weights_only: bool  # load weights only, reset step/optimizer (for SFT from pretrain)

    # logging
    log_every: int
    wandb_project: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(
            max_steps=d["max_steps"],
            grad_clip=d["grad_clip"],
            total_batch_size=d["total_batch_size"],
            optimizer=OptimizerConfig.from_dict(d["optimizer"]),
            schedule=ScheduleConfig.from_dict(d["schedule"]),
            eval_loss=EvalLossConfig.from_dict(d["eval_loss"]),
            eval_generate=EvalGenerateConfig.from_dict(d["eval_generate"]),
            eval_hellaswag=EvalHellaswagConfig.from_dict(d["eval_hellaswag"]),
            eval_gsm8k=EvalGSM8KConfig.from_dict(d["eval_gsm8k"]),
            checkpoint_dir=Path(d["checkpoint_dir"]),
            checkpoint_every=d["checkpoint_every"],
            resume_from=Path(d["resume_from"]) if d["resume_from"] is not None else None,
            resume_weights_only=d.get("resume_weights_only", False),
            log_every=d["log_every"],
            wandb_project=d["wandb_project"],
        )


@dataclass
class RunConfig:
    device: str
    device_type: str
    use_ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
