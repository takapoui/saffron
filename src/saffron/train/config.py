from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval import EvalGenerateConfig, EvalGSM8KConfig, EvalHellaswagConfig, EvalLossConfig


@dataclass
class SupervisedTrainConfig:
    # training loop
    max_steps: int
    grad_clip: float
    total_batch_size: int
    compile_model: bool

    # eval
    eval_loss: EvalLossConfig
    eval_generate: EvalGenerateConfig
    eval_hellaswag: EvalHellaswagConfig
    eval_gsm8k: EvalGSM8KConfig

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None
    resume_weights_only: bool

    # logging
    log_every: int
    wandb_project: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SupervisedTrainConfig:
        return cls(
            max_steps=d["max_steps"],
            grad_clip=d["grad_clip"],
            total_batch_size=d["total_batch_size"],
            compile_model=d.get("compile_model", False),
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
class RLTrainConfig:
    # training loop
    num_steps: int
    grad_clip: float
    compile_model: bool
    compile_ref_model: bool

    # rollout
    n_prompts_per_batch: int
    group_size: int
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float

    # loss
    clip_eps: float
    kl_coef: float

    # microbatch if a value <= B*G is used
    microbatch_size: int

    # eval
    eval_every: int | None
    eval_n_prompts: int

    # checkpointing
    checkpoint_dir: Path
    checkpoint_every: int
    resume_from: Path | None
    resume_weights_only: bool

    # logging
    log_every: int
    wandb_project: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RLTrainConfig:
        return cls(
            num_steps=d["num_steps"],
            grad_clip=d["grad_clip"],
            compile_model=d.get("compile_model", False),
            compile_ref_model=d.get("compile_ref_model", False),
            n_prompts_per_batch=d["n_prompts_per_batch"],
            group_size=d["group_size"],
            max_new_tokens=d["max_new_tokens"],
            temperature=d["temperature"],
            top_k=d["top_k"],
            top_p=d["top_p"],
            clip_eps=d["clip_eps"],
            kl_coef=d["kl_coef"],
            microbatch_size=d["microbatch_size"],
            eval_every=d["eval_every"],
            eval_n_prompts=d["eval_n_prompts"],
            checkpoint_dir=Path(d["checkpoint_dir"]),
            checkpoint_every=d["checkpoint_every"],
            resume_from=Path(d["resume_from"]) if d.get("resume_from") is not None else None,
            resume_weights_only=d.get("resume_weights_only", False),
            log_every=d["log_every"],
            wandb_project=d["wandb_project"],
        )
