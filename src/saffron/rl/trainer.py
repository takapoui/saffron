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
        rewards = [
            compute_reward(t, s)[0]
            for t, s in zip(rb.completion_texts, expanded_samples, strict=True)
        ]
        advs_lists = compute_grpo_advantages(rewards, rb.response_lens, group_size=cfg.group_size)

        T_pred = rb.input_ids.shape[1] - 1
        per_completion = torch.tensor([al[0] for al in advs_lists], dtype=torch.float32)
        advantages = per_completion.unsqueeze(-1).expand(-1, T_pred).to(device)

        with torch.no_grad():
            old_lp = compute_token_log_probs(
                self.model, rb.input_ids, rb.attention_mask, temperature=cfg.temperature
            )
            ref_lp = compute_token_log_probs(
                self.ref_model, rb.input_ids, rb.attention_mask, temperature=cfg.temperature
            )

        new_lp = compute_token_log_probs(
            self.model, rb.input_ids, rb.attention_mask, temperature=cfg.temperature
        )
        loss, loss_metrics = compute_grpo_loss(
            new_lp,
            old_lp,
            ref_lp,
            advantages,
            response_mask=rb.response_mask[:, 1:],
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
        )
        assert torch.isfinite(loss), f"non-finite loss at step {self.step}: {loss.item()}"

        loss.backward()  # type: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
        self.optimizer.step()  # type: ignore[reportUnknownMemberType]
        self.optimizer.zero_grad()

        dt = time.time() - t0
        reward_mean = sum(rewards) / len(rewards)
        return {
            "loss": loss.item(),
            "reward_mean": reward_mean,
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
        """Run rollouts on a fixed-seed slice of val data, return reward aggregates."""
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
            return {
                "eval/total_reward_mean": sum(totals) / n,
                "eval/format_reward_mean": sum(formats) / n,
                "eval/equation_reward_mean": sum(equations) / n,
                "eval/correct_rate": sum(1 for e in equations if e > 0) / n,
            }
        finally:
            self.model.train()
