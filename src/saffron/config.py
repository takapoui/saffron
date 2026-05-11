from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval import EvalGenerateConfig, EvalGSM8KConfig, EvalHellaswagConfig, EvalLossConfig


@dataclass
class TrainConfig:
    # optimization
    max_steps: int
    warmup_steps: int
    max_lr: float
    weight_decay: float
    grad_clip: float

    # data
    total_batch_size: int  # 524288 if cuda else 16384

    # eval
    eval_loss: EvalLossConfig
    eval_generate: EvalGenerateConfig
    eval_hellaswag: EvalHellaswagConfig
    eval_gsm8k: EvalGSM8KConfig

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None

    # logging
    log_every: int
    wandb_project: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(
            max_steps=d["max_steps"],
            warmup_steps=d["warmup_steps"],
            max_lr=d["max_lr"],
            weight_decay=d["weight_decay"],
            grad_clip=d["grad_clip"],
            total_batch_size=d["total_batch_size"],
            eval_loss=EvalLossConfig.from_dict(d["eval_loss"]),
            eval_generate=EvalGenerateConfig.from_dict(d["eval_generate"]),
            eval_hellaswag=EvalHellaswagConfig.from_dict(d["eval_hellaswag"]),
            eval_gsm8k=EvalGSM8KConfig.from_dict(d["eval_gsm8k"]),
            checkpoint_dir=Path(d["checkpoint_dir"]),
            checkpoint_every=d["checkpoint_every"],
            resume_from=Path(d["resume_from"]) if d["resume_from"] is not None else None,
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
