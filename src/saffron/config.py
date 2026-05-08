from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    block_size: int
    n_layer: int
    n_head: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(
            vocab_size=d["vocab_size"],
            n_embd=d["n_embd"],
            block_size=d["block_size"],
            n_layer=d["n_layer"],
            n_head=d["n_head"],
        )


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
    tokenizer: str

    # eval
    eval_loss_every: int
    eval_loss_steps: int  # how many val batches to average over
    eval_generate_every: int
    eval_hellaswag_every: int

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
            tokenizer=d["tokenizer"],
            eval_loss_every=d["eval_loss_every"],
            eval_loss_steps=d["eval_loss_steps"],
            eval_generate_every=d["eval_generate_every"],
            eval_hellaswag_every=d["eval_hellaswag_every"],
            checkpoint_dir=Path(d["checkpoint_dir"]),
            checkpoint_every=d["checkpoint_every"],
            resume_from=Path(d["resume_from"]) if d["resume_from"] is not None else None,
            log_every=d["log_every"],
            wandb_project=d["wandb_project"],
        )


@dataclass
class DataConfig:
    data_root: Path
    batch_size: int
    seq_len: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(
            data_root=Path(d["data_root"]),
            batch_size=d["batch_size"],
            seq_len=d["seq_len"],
        )


@dataclass
class PrepConfig:
    data_root: Path
    dataset: str
    name: str
    shard_size: int
    tokenizer: str
    dataset_split: str
    num_validation_shards: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrepConfig:
        return cls(
            data_root=Path(d["data_root"]),
            dataset=d["dataset"],
            name=d["name"],
            shard_size=d["shard_size"],
            tokenizer=d["tokenizer"],
            dataset_split=d["dataset_split"],
            num_validation_shards=d["num_validation_shards"],
        )


@dataclass
class RunConfig:
    device: str
    device_type: str
    use_ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
