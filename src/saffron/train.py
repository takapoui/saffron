from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from typing import cast

import tiktoken
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

from .config import RunConfig, TrainConfig
from .dataloader import DataLoader
from .hellaswag import evaluate_hellaswag
from .helpers import get_peak_flops
from .model import Model
from .optim import get_lr_cosine

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.AdamW,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config: TrainConfig,
        run_config: RunConfig,
    ) -> None:
        if run_config.use_ddp:
            self.raw_model = cast(Model, torch.compile(model.to(run_config.device)))  # pyright: ignore[reportUnknownMemberType]
            self.model = DistributedDataParallel(
                self.raw_model, device_ids=[run_config.ddp_local_rank]
            )
        else:
            self.raw_model = cast(Model, torch.compile(model.to(run_config.device)))  # pyright: ignore[reportUnknownMemberType]
            self.model = self.raw_model

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_config = train_config
        self.run_config = run_config
        self.enc = tiktoken.get_encoding(train_config.tokenizer)
        self._num_model_parameters = sum(p.numel() for p in model.parameters())
        self._device_peak_flops = get_peak_flops(run_config.device_type)

        # master process
        self.master_process = run_config.ddp_rank == 0

        B, T = train_loader.B, train_loader.T
        assert train_config.total_batch_size % (B * T * run_config.ddp_world_size) == 0, (
            "total_batch_size must be divisible by B * T * world_size"
        )
        self.accumulation_steps = train_config.total_batch_size // (
            B * T * run_config.ddp_world_size
        )
        if self.master_process:
            logger.info("Using gradient accumulation over %d steps.", self.accumulation_steps)

        resume_from = train_config.resume_from
        if resume_from is None:
            self.step = 0
            self.train_loader.reset()
        else:
            checkpoint = torch.load(resume_from, weights_only=False, map_location=run_config.device)
            self.raw_model.load_state_dict(checkpoint["model_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.step = checkpoint["step"] + 1
            self.train_loader.reset()
            self.train_loader.advance(checkpoint["step"] * train_config.total_batch_size)

        self.tokens_seen = self.step * train_config.total_batch_size

        self.use_wandb = self.master_process and train_config.wandb_project is not None
        if self.use_wandb:
            wandb.init(project=train_config.wandb_project, config=dataclasses.asdict(train_config))
            wandb.define_metric("tokens_seen")
            wandb.define_metric("*", step_metric="tokens_seen")

    def train(self) -> None:
        self.model.train()
        for step in range(self.step, self.train_config.max_steps):
            if step % self.train_config.eval_loss_every == 0:
                metrics = {"eval_loss": self._eval_loss()}
                self._log(step, metrics)

            if step % self.train_config.eval_generate_every == 0:
                self._eval_generate_task(step=step)

            if step % self.train_config.eval_hellaswag_every == 0:
                metrics = self._eval_hellaswag_task(step=step)
                if metrics:
                    self._log(step, metrics)

            if self.master_process and step % self.train_config.checkpoint_every == 0:
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
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.grad_clip
            )
            lr = get_lr_cosine(
                step=step,
                max_steps=self.train_config.max_steps,
                max_lr=self.train_config.max_lr,
                warmup_steps=self.train_config.warmup_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()  # type: ignore[reportUnknownMemberType]

            if self.run_config.device_type == "cuda":
                torch.cuda.synchronize()
            elif self.run_config.device_type == "mps":
                torch.mps.synchronize()
            t1 = time.time()
            self.tokens_seen += self.train_config.total_batch_size
            tok_per_sec = self.train_config.total_batch_size / (t1 - t0)
            mfu = (
                6
                * self._num_model_parameters
                * tok_per_sec
                / (self._device_peak_flops * self.run_config.ddp_world_size)
            )
            metrics = {
                "sec": t1 - t0,
                "norm": norm.item(),
                "lr": lr,
                "loss": loss_accum,
                "tok_per_sec": tok_per_sec,
                "mfu": mfu,
                "tokens_seen": self.tokens_seen,
            }
            if step % self.train_config.log_every == 0:
                self._log(step, metrics)
        if self.use_wandb:
            wandb.finish()

    def _eval_loss(self) -> float:
        self.val_loader.reset()
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.train_config.eval_loss_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.run_config.device), y.to(self.run_config.device)
                with torch.autocast(device_type=self.run_config.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                loss /= self.train_config.eval_loss_steps
                val_loss_accum += loss.item()
            if self.run_config.use_ddp:
                val_loss_tensor = torch.tensor(val_loss_accum, device=self.run_config.device)
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)  # type: ignore[reportUnknownMemberType]
                val_loss_accum = val_loss_tensor.item()
        self.model.train()
        return val_loss_accum

    def _eval_generate_task(self, step: int) -> None:
        if self.master_process:
            prompt = "Today is a nice day, because"
            idx = torch.tensor(self.enc.encode_ordinary(prompt), dtype=torch.long)
            idx = idx.unsqueeze(0).repeat(5, 1)
            idx = idx.to(self.run_config.device)
            tokens = self.raw_model.generate(idx=idx, max_new_tokens=50)
            completions = [self.enc.decode(tokens[i, :].tolist()) for i in range(tokens.shape[0])]  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            for sample in completions:
                logger.info(f"Step {step} sample: {sample}")
            if self.use_wandb:
                table = wandb.Table(columns=["step", "completion"])
                for sample in completions:
                    table.add_data(step, sample)  # type: ignore[reportUnknownMemberType]
                wandb.log({"sample": table}, step=step)

    def _eval_hellaswag_task(self, step: int) -> dict[str, float]:
        if self.master_process:
            metrics = {
                "hellaswag": evaluate_hellaswag(
                    self.raw_model,
                    self.run_config.device,
                    self.run_config.device_type,
                    enc=self.enc,
                )
            }

            return metrics
        return {}

    def _save_checkpoint(self, step: int, keep_last: int = 3) -> None:
        self.train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.train_config.checkpoint_dir / f"ckpt_{step:06d}.pt"
        obj = {
            "step": step,
            "model_dict": self.raw_model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "model_config": self.raw_model.config,
            "train_config": self.train_config,
        }
        torch.save(obj, path)
        logger.info(f"Saved checkpoint to {path}")

        checkpoints = sorted(self.train_config.checkpoint_dir.glob("ckpt_*.pt"))
        for old in checkpoints[:-keep_last]:
            old.unlink()
            logger.info(f"Deleted old checkpoint {old}")

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        if self.master_process:
            info = [f"step: {step:5d}"] + [f"{key}: {val:.4f}" for key, val in metrics.items()]
            logger.info(" | ".join(info))

        if self.use_wandb:
            wandb.log({"tokens_seen": self.tokens_seen, **metrics}, step=step)
