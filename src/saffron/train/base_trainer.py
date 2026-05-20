from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import torch
import wandb

from ..helpers import RunConfig, format_metric_line, init_wandb
from ..model import BaseModel
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Shared infrastructure for trainers: device placement, compile, wandb, logging.

    Subclasses own their own train loop and step logic. The base only provides
    helpers for bells and whistles — it does not orchestrate.
    """

    def __init__(self, run_config: RunConfig, *, checkpoint_dir: Path | None = None) -> None:
        self.run_config = run_config
        self.master_process = run_config.ddp_rank == 0
        self._train_start_time = time.time()
        self.use_wandb = False
        self._sample_rows: list[list[Any]] = []
        self.checkpoint_dir = checkpoint_dir

    def _prepare_model(self, model: BaseModel, *, compile: bool) -> BaseModel:
        """Move to device and (optionally) torch.compile."""
        model = model.to(self.run_config.device)
        if compile and model.supports_compile(self.run_config.device_type):
            model = cast(BaseModel, torch.compile(model))  # pyright: ignore[reportUnknownMemberType]
        return model

    def _prepare_frozen_model(self, model: BaseModel, *, compile: bool) -> BaseModel:
        """Move to device, eval(), requires_grad_(False), and (optionally) torch.compile."""
        model = model.to(self.run_config.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if compile and model.supports_compile(self.run_config.device_type):
            model = cast(BaseModel, torch.compile(model))  # pyright: ignore[reportUnknownMemberType]
        return model

    def _validate_tokenizer_match(
        self, model_tokenizer: Tokenizer, loader_tokenizer: Tokenizer
    ) -> None:
        if model_tokenizer.name != loader_tokenizer.name:
            raise ValueError(
                f"Model tokenizer '{model_tokenizer.name}' does not match "
                f"data tokenizer '{loader_tokenizer.name}'. "
                "Re-run data prep with the correct tokenizer."
            )

    def _init_wandb(
        self,
        project: str | None,
        config_dict: dict[str, Any],
        *,
        step_metric: str | None = None,
    ) -> None:
        """Set up wandb on the master process. No-op otherwise.

        Sets self.use_wandb so subclasses can gate further wandb calls.
        """
        self.use_wandb = self.master_process and project is not None
        if self.use_wandb:
            assert project is not None
            init_wandb(project, config_dict, step_metric=step_metric)

    def _log(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        extra_wandb: dict[str, Any] | None = None,
    ) -> None:
        """Log metrics to the file logger (master only) and wandb (if enabled).

        extra_wandb is merged into the wandb payload but not into the log line —
        used by SFT to inject tokens_seen without polluting metric names.
        """
        if self.master_process:
            logger.info(format_metric_line(step, metrics))
        if self.use_wandb:
            payload: dict[str, Any] = dict(metrics)
            if extra_wandb:
                payload.update(extra_wandb)
            wandb.log(payload, step=step)

    def _log_table(
        self,
        name: str,
        columns: list[str],
        rows: list[list[Any]],
        *,
        step: int,
        extra_wandb: dict[str, Any] | None = None,
    ) -> None:
        if not self.use_wandb:
            return
        table = wandb.Table(columns=columns, data=rows)
        payload: dict[str, Any] = {name: table}
        if extra_wandb:
            payload.update(extra_wandb)
        wandb.log(payload, step=step)

    def _finish_wandb(self) -> None:
        if self.use_wandb:
            wandb.finish()

    def _write_checkpoint(
        self,
        step: int,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        *,
        extra: dict[str, Any] | None = None,
        keep_last: int = 3,
    ) -> None:
        """Write checkpoint and prune older ones. Caller passes config-specific extras."""
        assert self.checkpoint_dir is not None, "checkpoint_dir not configured on BaseTrainer"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"ckpt_{step:06d}.pt"
        payload: dict[str, Any] = {
            "step": step,
            "model_dict": model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "model_config": model.config,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info(f"Saved checkpoint to {path}")

        checkpoints = sorted(self.checkpoint_dir.glob("ckpt_*.pt"))
        for old in checkpoints[:-keep_last]:
            old.unlink()
            logger.info(f"Deleted old checkpoint {old}")

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            torch.load(path, weights_only=False, map_location=self.run_config.device),
        )
