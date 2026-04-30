from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .dataloader import DataLoader
from .model import Model


@dataclass
class TrainConfig:
    # optimization
    max_steps: int
    warmup_steps: int
    max_lr: float
    weight_decay: float
    grad_clip: float

    # data
    batch_size: int
    seq_len: int

    # eval
    eval_every: int
    eval_steps: int  # how many val batches to average over

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None

    # logging
    log_every: int
    wandb_project: str | None


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.AdamW,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ) -> None:
        raise NotImplementedError

    def train(self) -> None:
        # main loop: lr schedule → forward → backward → clip → step
        # calls _eval and _save_checkpoint at intervals
        raise NotImplementedError

    def _eval_loss(self) -> float:
        # average loss over eval_steps val batches
        raise NotImplementedError

    def _eval_tasks(self) -> dict[str, float]:
        # external evals: HellaSwag, etc.
        # returns {"hellaswag": 0.42, ...}
        raise NotImplementedError

    def _save_checkpoint(self, step: int) -> None:
        # saves model + optimizer state_dict + step + config
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, path: Path) -> Trainer:
        # reconstructs model, optimizer, loader, resumes from step
        raise NotImplementedError

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        # logger.info + optional wandb
        raise NotImplementedError
