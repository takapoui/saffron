"""GRPO trainer. Mirrors the structure of saffron.train.Trainer."""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import torch

from ..config import RLConfig, RunConfig
from ..data import RLDataLoader
from ..eval.rl.reward import compute_reward
from ..helpers import format_metric_line, init_wandb
from ..model import BaseModel
from .advantage import compute_grpo_advantages
from .logprobs import compute_token_log_probs
from .loss import compute_grpo_loss
from .rollout import rollout

logger = logging.getLogger(__name__)


class RLTrainer:
    def __init__(
        self,
        model: BaseModel,
        ref_model: BaseModel,
        optimizer: torch.optim.AdamW,
        train_loader: RLDataLoader,
        val_loader: RLDataLoader,
        rl_config: RLConfig,
        run_config: RunConfig,
    ) -> None:
        if run_config.use_ddp:
            raise ValueError("RLTrainer does not support DDP yet")

        tokenizer = model.tokenizer
        self.model = model.to(run_config.device)
        self.tokenizer = tokenizer

        # Reference model: frozen, eval-mode, no grad tracking.
        ref_model = ref_model.to(run_config.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rl_config = rl_config
        self.run_config = run_config

        if self.tokenizer.name != train_loader.tokenizer.name:
            raise ValueError(
                f"Model tokenizer '{self.tokenizer.name}' does not match "
                f"data tokenizer '{train_loader.tokenizer.name}'. "
                "Re-run data prep with the correct tokenizer."
            )

        self.master_process = run_config.ddp_rank == 0
        self.step = 0
        self._rng = np.random.default_rng(42)
        self._train_start_time = time.time()

        self.use_wandb = self.master_process and rl_config.wandb_project is not None
        if self.use_wandb:
            assert rl_config.wandb_project is not None
            init_wandb(rl_config.wandb_project, dataclasses.asdict(rl_config))
            self._sample_rows: list[list[object]] = []

    def train(self) -> None:
        self.model.train()
        for step in range(self.step, self.rl_config.num_steps):
            self.step = step
            if self._should_eval(step):
                self._log(step, self._evaluate())
            metrics = self._step()
            if step % self.rl_config.log_every == 0:
                self._log(step, metrics)
        # Final eval at end so the last checkpoint has a corresponding eval point.
        if self.rl_config.eval_every is not None:
            self._log(self.rl_config.num_steps, self._evaluate())
        if self.use_wandb:
            import wandb

            wandb.finish()

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

        total_rows = rb.input_ids.shape[0]
        mb_size = max(1, min(cfg.microbatch_size, total_rows))
        with torch.no_grad():
            old_lp = compute_token_log_probs(
                self.model, rb.input_ids, rb.attention_mask, temperature=cfg.temperature
            )
            ref_lp = compute_token_log_probs(
                self.ref_model, rb.input_ids, rb.attention_mask, temperature=cfg.temperature
            )

        # Gradients accumulate across microbatches; one optimizer.step() at the end.
        loss_value = 0.0
        loss_metrics: dict[str, float] = {}

        self.optimizer.zero_grad()
        for start in range(0, total_rows, mb_size):
            end = min(start + mb_size, total_rows)
            k = end - start
            weight = k / total_rows  # so accumulated grad ≈ full-batch grad

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
            )
            assert torch.isfinite(loss_chunk), (
                f"non-finite loss at step {self.step}, microbatch [{start}:{end}]: "
                f"{loss_chunk.item()}"
            )

            (loss_chunk * weight).backward()  # type: ignore[reportUnknownMemberType]
            loss_value += loss_chunk.item() * weight
            for key, val in metrics_chunk.items():
                loss_metrics[key] = loss_metrics.get(key, 0.0) + val * weight

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
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
            **loss_metrics,
        }

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        if self.master_process:
            logger.info(format_metric_line(step, metrics))
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

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
                import wandb

                sample_rng = np.random.default_rng(self.step)
                indices: list[int] = sample_rng.choice(n, size=min(5, n), replace=False).tolist()
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
                table = wandb.Table(
                    columns=[
                        "step",
                        "nums",
                        "target",
                        "completion",
                        "total_reward",
                        "format_reward",
                        "equation_reward",
                    ],
                    data=self._sample_rows,
                )
                wandb.log({"eval_samples": table}, step=self.step)

            return {
                "eval_total_reward_mean": sum(totals) / n,
                "eval_format_reward_mean": sum(formats) / n,
                "eval_equation_reward_mean": sum(equations) / n,
                "eval_correct_rate": sum(1 for e in equations if e > 0) / n,
            }
        finally:
            self.model.train()
