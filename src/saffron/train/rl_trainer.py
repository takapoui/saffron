"""GRPO trainer. Mirrors the structure of SupervisedTrainer."""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import torch

from ..data import RLDataLoader
from ..eval.rl.reward import compute_reward
from ..helpers import RunConfig
from ..model import BaseModel
from ..optim import OptimizerConfig, get_lr
from ..rl.advantage import compute_grpo_advantages
from ..rl.logprobs import compute_token_log_probs
from ..rl.loss import compute_grpo_loss
from ..rl.rollout import rollout
from .base_trainer import BaseTrainer
from .config import RLTrainConfig

logger = logging.getLogger(__name__)


class RLTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        ref_model: BaseModel,
        optimizer: torch.optim.Optimizer,
        optimizer_config: OptimizerConfig,
        train_loader: RLDataLoader,
        val_loader: RLDataLoader,
        rl_config: RLTrainConfig,
        run_config: RunConfig,
    ) -> None:
        super().__init__(run_config, checkpoint_dir=rl_config.checkpoint_dir)
        if run_config.use_ddp:
            raise ValueError("RLTrainer does not support DDP yet")

        self.tokenizer = model.tokenizer
        self._validate_tokenizer_match(self.tokenizer, train_loader.tokenizer)

        self.raw_model = self._prepare_model(model, compile=rl_config.compile_model)
        self.model = self.raw_model  # DDP wrap point when RL learns DDP
        self.ref_model = self._prepare_frozen_model(ref_model, compile=rl_config.compile_ref_model)

        self.optimizer = optimizer
        self.optimizer_config = optimizer_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rl_config = rl_config

        self.step = 0
        self._rng = np.random.default_rng(42)

        resume_from = rl_config.resume_from
        if resume_from is not None:
            checkpoint = self._load_checkpoint(resume_from)
            self.raw_model.load_state_dict(checkpoint["model_dict"])
            if rl_config.resume_weights_only:
                logger.info("Loaded weights from %s (weights only, step reset to 0).", resume_from)
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
                self.step = checkpoint["step"] + 1
                if "rng_state" in checkpoint:
                    self._rng.bit_generator.state = checkpoint["rng_state"]
                logger.info("Resumed from %s (step %d, rng restored).", resume_from, self.step)

        self._init_wandb(
            rl_config.wandb_project,
            {**dataclasses.asdict(rl_config), **dataclasses.asdict(optimizer_config)},
        )

    def train(self) -> None:
        # Eval mode disables dropout so old_lp and new_lp_chunk see identical forwards
        # at equal weights. With dropout on, different masks would corrupt the PPO ratio.
        self.model.eval()
        for step in range(self.step, self.rl_config.num_steps):
            self.step = step
            lr = get_lr(step, self.rl_config.num_steps, self.optimizer_config)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if self._should_eval(step):
                self._log(step, self._evaluate())
            metrics = self._step()
            if step % self.rl_config.log_every == 0:
                self._log(step, {"lr": lr, **metrics})
            # Save after _step so checkpoint at step N reflects state through step N
            # (resume from N+1 picks up correctly).
            if self.master_process and step % self.rl_config.checkpoint_every == 0:
                self._save_checkpoint(step)
        # Final eval and checkpoint at end so the last checkpoint has a corresponding eval point.
        last_step = self.rl_config.num_steps
        if self.rl_config.eval_every is not None:
            self._log(last_step, self._evaluate())
        if self.master_process and (last_step - 1) % self.rl_config.checkpoint_every != 0:
            self._save_checkpoint(last_step - 1)
        self._finish_wandb()

    def _save_checkpoint(self, step: int) -> None:
        self._write_checkpoint(
            step,
            self.raw_model,
            self.optimizer,
            extra={
                "rl_config": self.rl_config,
                "rng_state": self._rng.bit_generator.state,
            },
        )

    def _should_eval(self, step: int) -> bool:
        return self.rl_config.eval_every is not None and step % self.rl_config.eval_every == 0

    def _step(self) -> dict[str, float]:
        cfg = self.rl_config
        device = self.run_config.device
        t0 = time.time()

        batch = self.train_loader.sample_batch(n=cfg.n_prompts_per_batch, rng=self._rng)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        rb = rollout(
            self.model,
            input_ids,
            attention_mask,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        expanded_samples = [s for s in batch.samples for _ in range(cfg.group_size)]
        reward_results = [
            compute_reward(t, s) for t, s in zip(rb.completion_texts, expanded_samples, strict=True)
        ]
        rewards = [r for r, _ in reward_results]
        format_rewards = [m["format_reward"] for _, m in reward_results]
        equation_rewards = [m["equation_reward"] for _, m in reward_results]
        advs_lists = compute_grpo_advantages(rewards, rb.response_lens, group_size=cfg.group_size)

        T_pred = rb.input_ids.shape[1] - 1
        per_completion = torch.tensor([al[0] for al in advs_lists], dtype=torch.float32)
        advantages = per_completion.unsqueeze(-1).expand(-1, T_pred).to(device)
        zero_advantage_ratio = (per_completion.abs() < 1e-6).float().mean().item()

        total_rows = rb.input_ids.shape[0]
        mb_size = max(1, min(cfg.microbatch_size, total_rows))
        # Shared full-batch denominator so microbatches accumulate the exact full-batch gradient.
        total_response_len = rb.response_mask[:, 1:].sum().clamp(min=1)
        with torch.no_grad():
            old_lp = torch.cat(
                [
                    compute_token_log_probs(
                        self.model,
                        rb.input_ids[s : s + mb_size],
                        rb.attention_mask[s : s + mb_size],
                        temperature=cfg.temperature,
                    )
                    for s in range(0, total_rows, mb_size)
                ]
            )
            ref_lp = torch.cat(
                [
                    compute_token_log_probs(
                        self.ref_model,
                        rb.input_ids[s : s + mb_size],
                        rb.attention_mask[s : s + mb_size],
                        temperature=cfg.temperature,
                    )
                    for s in range(0, total_rows, mb_size)
                ]
            )

        # Gradients accumulate across microbatches; one optimizer.step() at the end.
        loss_value = 0.0
        loss_metrics: dict[str, float] = {}

        self.optimizer.zero_grad()
        for start in range(0, total_rows, mb_size):
            end = min(start + mb_size, total_rows)

            new_lp_chunk = compute_token_log_probs(
                self.model,
                rb.input_ids[start:end],
                rb.attention_mask[start:end],
                temperature=cfg.temperature,
            )
            loss_chunk, metrics_chunk = compute_grpo_loss(
                new_lp_chunk,
                old_lp[start:end],
                ref_lp[start:end],
                advantages[start:end],
                response_mask=rb.response_mask[start:end, 1:],
                clip_eps=cfg.clip_eps,
                kl_coef=cfg.kl_coef,
                total_response_len=total_response_len,
            )
            assert torch.isfinite(loss_chunk), (
                f"non-finite loss at step {self.step}, microbatch [{start}:{end}]: "
                f"{loss_chunk.item()}"
            )

            # Each microbatch's loss is already normalized by the full-batch total_response_len,
            # so backward() accumulates the exact full-batch gradient without extra weighting.
            loss_chunk.backward()  # type: ignore[reportUnknownMemberType]
            loss_value += loss_chunk.item()
            for key, val in metrics_chunk.items():
                loss_metrics[key] = loss_metrics.get(key, 0.0) + val

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=cfg.grad_clip
        ).item()
        self.optimizer.step()  # type: ignore[reportUnknownMemberType]
        self.optimizer.zero_grad()

        dt = time.time() - t0
        n_completions = len(rewards)
        reward_mean = sum(rewards) / n_completions
        format_reward_mean = sum(format_rewards) / n_completions
        equation_reward_mean = sum(equation_rewards) / n_completions
        avg_response_len = sum(rb.response_lens) / n_completions

        # Truncation: completion used the full generation budget. It likely means the model
        # didn't emit a stop toke. (Small chance that it produced exactly that number of tokens.)
        truncation_rate = (
            sum(1 for L in rb.response_lens if cfg.max_new_tokens == L) / n_completions
        )

        return {
            "loss": loss_value,
            "reward_mean": reward_mean,
            "format_reward_mean": format_reward_mean,
            "equation_reward_mean": equation_reward_mean,
            "avg_response_len": avg_response_len,
            "truncation_rate": truncation_rate,
            "step_time": dt,
            "zero_advantage_ratio": zero_advantage_ratio,
            "grad_norm": grad_norm,
            **loss_metrics,
        }

    def _evaluate(self) -> dict[str, float]:
        """Run rollouts on a fixed-seed slice of val data, return reward aggregates.

        Also accumulates a few sample completions into self._sample_rows so the
        wandb table grows over training and you can see model behavior evolve.
        """
        cfg = self.rl_config
        device = self.run_config.device
        self.model.eval()
        try:
            # Fixed seed so the same val prompts are scored each eval round.
            rng = np.random.default_rng(0)
            batch = self.val_loader.sample_batch(n=cfg.eval_n_prompts, rng=rng)
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)

            rb = rollout(
                self.model,
                input_ids,
                attention_mask,
                group_size=cfg.group_size,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            expanded_samples = [s for s in batch.samples for _ in range(cfg.group_size)]
            scored = [
                compute_reward(t, s)
                for t, s in zip(rb.completion_texts, expanded_samples, strict=True)
            ]
            totals = [r for r, _ in scored]
            formats = [m["format_reward"] for _, m in scored]
            equations = [m["equation_reward"] for _, m in scored]
            n = len(scored)

            # Log a few sample completions to a wandb table per eval round.
            if self.use_wandb:
                sample_rng = np.random.default_rng(self.step)
                indices: list[int] = sample_rng.choice(n, size=min(100, n), replace=False).tolist()
                for i in indices:
                    self._sample_rows.append(
                        [
                            self.step,
                            str(expanded_samples[i]["nums"]),
                            expanded_samples[i]["target"],
                            rb.completion_texts[i],
                            totals[i],
                            formats[i],
                            equations[i],
                        ]
                    )
                self._log_table(
                    "eval_samples",
                    columns=[
                        "step",
                        "nums",
                        "target",
                        "completion",
                        "total_reward",
                        "format_reward",
                        "equation_reward",
                    ],
                    rows=self._sample_rows,
                    step=self.step,
                )

            return {
                "eval_total_reward_mean": sum(totals) / n,
                "eval_format_reward_mean": sum(formats) / n,
                "eval_equation_reward_mean": sum(equations) / n,
                "eval_correct_rate": sum(1 for e in equations if e > 0) / n,
            }
        finally:
            self.model.eval()
