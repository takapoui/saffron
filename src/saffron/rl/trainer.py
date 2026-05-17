"""RL pipeline smoke test: 5 GRPO iterations on Qwen-0.5B-Instruct / Countdown."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from saffron.data import RLDataLoader
from saffron.eval.rl.reward import compute_reward
from saffron.helpers import get_default_device
from saffron.model import HFConfig, HFModel
from saffron.rl.advantage import compute_grpo_advantages
from saffron.rl.logprobs import compute_token_log_probs
from saffron.rl.loss import compute_grpo_loss
from saffron.rl.rollout import rollout


def train() -> None:
    device = get_default_device()
    print(f"device: {device}")

    model = HFModel(HFConfig.from_dict({"hf_model_name": "Qwen/Qwen2.5-0.5B"})).to(device)
    # model.hf_model.gradient_checkpointing_enable()

    ref_model = HFModel(HFConfig.from_dict({"hf_model_name": "Qwen/Qwen2.5-0.5B"})).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dataloader = RLDataLoader(
        path=Path("data/rl/countdown/train.jsonl"),
        tokenizer=model.tokenizer,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    rng = np.random.default_rng(42)

    group_size = 4
    n = 1

    for step in range(200):
        t0 = time.time()
        batch = dataloader.sample_batch(n=n, rng=rng)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        rb = rollout(
            model,
            input_ids,
            attention_mask,
            group_size=group_size,
            max_new_tokens=200,
            temperature=1.0,
        )

        expanded_samples = [s for s in batch.samples for _ in range(group_size)]
        rewards = [
            compute_reward(t, s)[0]
            for t, s in zip(rb.completion_texts, expanded_samples, strict=True)
        ]
        advs_lists = compute_grpo_advantages(rewards, rb.response_lens, group_size=group_size)

        T_pred = rb.input_ids.shape[1] - 1
        per_completion = torch.tensor([al[0] for al in advs_lists], dtype=torch.float32)
        advantages = per_completion.unsqueeze(-1).expand(-1, T_pred).to(device)

        with torch.no_grad():
            old_lp = compute_token_log_probs(
                model, rb.input_ids, rb.attention_mask, temperature=1.0
            )
            ref_lp = compute_token_log_probs(
                ref_model, rb.input_ids, rb.attention_mask, temperature=1.0
            )

        new_lp = compute_token_log_probs(model, rb.input_ids, rb.attention_mask, temperature=1.0)
        loss, metrics = compute_grpo_loss(
            new_lp,
            old_lp,
            ref_lp,
            advantages,
            response_mask=rb.response_mask[:, 1:],
            clip_eps=0.2,
            kl_coef=0.05,
        )

        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss.item()}"

        loss.backward()  # type: ignore[reportUnknownMemberType]
        optimizer.step()  # type: ignore[reportUnknownMemberType]
        optimizer.zero_grad()

        dt = time.time() - t0
        reward_mean = sum(rewards) / len(rewards)
        print(
            f"step {step}: {dt:.1f}s | loss={loss.item():.4f} | "
            f"reward_mean={reward_mean:.2f} | response_lens={rb.response_lens} | "
            f"metrics={metrics}"
        )
        print(f"  sample completion: {rb.completion_texts[0][:160]!r}")
