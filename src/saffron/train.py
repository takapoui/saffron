from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from .dataloader import DataLoader
from .model import Model
from .optim import get_lr_cosine

logger = logging.getLogger(__name__)


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
    eval_loss_every: int
    eval_loss_steps: int  # how many val batches to average over
    eval_task_every: int

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None

    # logging
    log_every: int
    wandb_project: str | None


@dataclass
class RunConfig:
    device: str
    device_type: str
    use_ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.AdamW,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        run_config: RunConfig,
    ) -> None:
        if run_config.use_ddp:
            self.raw_model = model.to(run_config.device)
            self.model = DistributedDataParallel(
                self.raw_model, device_ids=[run_config.ddp_local_rank]
            )
        else:
            self.raw_model = model.to(run_config.device)
            self.model = self.raw_model

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_config = run_config

        # master process
        self.master_process = run_config.ddp_rank == 0

        B, T = train_loader.B, train_loader.T
        assert config.total_batch_size % (B * T * run_config.ddp_world_size) == 0, (
            "total_batch_size must be divisible by B * T * world_size"
        )
        self.accumulation_steps = config.total_batch_size // (B * T * run_config.ddp_world_size)
        if self.master_process:
            logger.info("Using gradient accumulation over %d steps.", self.accumulation_steps)

        # reset optimizer parameters
        if self.config.resume_from is None:
            self.step = 0
            self.train_loader.reset()
            self.val_loader.reset()
        else:
            raise NotImplementedError

    def train(self) -> None:
        self.model.train()
        for step in range(self.step, self.config.max_steps):
            if step % self.config.eval_loss_every == 0:
                metrics = {"eval_loss": self._eval_loss()}
                self._log(step, metrics)

            if step % self.config.eval_task_every == 0:
                metrics = self._eval_tasks()
                if metrics:
                    self._log(step, metrics)

            if self.master_process and step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)

            t0 = time.time()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.accumulation_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.run_config.device), y.to(self.run_config.device)
                with torch.autocast(device_type=self.run_config.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                loss /= self.accumulation_steps
                loss_accum += loss.item()

                is_last_step = micro_step == self.accumulation_steps - 1
                sync_gradients = not self.run_config.use_ddp or is_last_step
                if not sync_gradients:
                    assert isinstance(self.model, DistributedDataParallel)
                    ctx = self.model.no_sync()
                else:
                    ctx = contextlib.nullcontext()
                with ctx:
                    loss.backward()
            if self.run_config.use_ddp:
                loss_tensor = torch.tensor(loss_accum, device=self.run_config.device)
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)  # type: ignore[reportUnknownMemberType]
                loss_accum = loss_tensor.item()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            lr = get_lr_cosine(
                step=step,
                max_steps=self.config.max_steps,
                max_lr=self.config.max_lr,
                warmup_steps=self.config.warmup_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()  # type: ignore[reportUnknownMemberType]

            if self.run_config.device_type == "cuda":
                torch.cuda.synchronize()
            elif self.run_config.device_type == "mps":
                torch.mps.synchronize()
            t1 = time.time()
            metrics = {
                "sec": t1 - t0,
                "norm": norm.item(),
                "lr": lr,
                "loss": loss_accum,
                "tok/sec": self.config.total_batch_size / (t1 - t0),
            }
            if step % self.config.log_every == 0:
                self._log(step, metrics)

    def _eval_loss(self) -> float:
        self.val_loader.reset()
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.config.eval_loss_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.run_config.device), y.to(self.run_config.device)
                with torch.autocast(device_type=self.run_config.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                loss /= self.config.eval_loss_steps
                val_loss_accum += loss.item()
            if self.run_config.use_ddp:
                val_loss_tensor = torch.tensor(val_loss_accum, device=self.run_config.device)
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)  # type: ignore[reportUnknownMemberType]
                val_loss_accum = val_loss_tensor.item()
        self.model.train()
        return val_loss_accum

    def _eval_tasks(self) -> dict[str, float]:
        # returns {"hellaswag": 0.42, ...}
        return {}

    def _save_checkpoint(self, step: int) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"ckpt_{step:06d}.pt"
        obj = {
            "step": step,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": self.raw_model.config,
            "train_config": self.config,
            "run_config": self.run_config,
        }
        torch.save(obj, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def from_checkpoint(cls, path: Path) -> Trainer:
        # reconstructs model, optimizer, loader, resumes from step
        # Don't forget to advance data loader to step * total_batch_size
        raise NotImplementedError

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        if self.master_process:
            info = [f"step: {step:5d}"] + [f"{key}: {val:.4f}" for key, val in metrics.items()]
            logger.info(" | ".join(info))

        # TODO wandb
