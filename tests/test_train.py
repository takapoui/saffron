"""Tests for Trainer scheduling and resume logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from saffron.config import RunConfig, TrainConfig
from saffron.eval import EvalGenerateConfig, EvalGSM8KConfig, EvalHellaswagConfig, EvalLossConfig
from saffron.model import GPT2, GPT2Config
from saffron.train import Trainer

# ---------------------------------------------------------------------------
# Helpers shared across trainer tests
# ---------------------------------------------------------------------------


def _run_config() -> RunConfig:
    return RunConfig(
        device="cpu",
        device_type="cpu",
        use_ddp=False,
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
    )


def _train_config(
    tmp_path: Path,
    *,
    max_steps: int,
    eval_loss_every: int,
    checkpoint_every: int = 1000,
    gsm8k_every: int | None = None,
    resume_from: Path | None = None,
    resume_weights_only: bool = False,
    total_batch_size: int = 4,
) -> TrainConfig:
    return TrainConfig(
        max_steps=max_steps,
        warmup_steps=1,
        max_lr=1e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        total_batch_size=total_batch_size,
        eval_loss=EvalLossConfig(every=eval_loss_every, steps=1),
        eval_generate=EvalGenerateConfig(
            every=None, prompt="hi", samples=1, max_tokens=5, use_chat_template=False
        ),
        eval_hellaswag=EvalHellaswagConfig(every=None),
        eval_gsm8k=EvalGSM8KConfig(every=gsm8k_every),
        checkpoint_dir=tmp_path / "ckpt",
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
        resume_weights_only=resume_weights_only,
        log_every=10_000,
        wandb_project=None,
    )


@pytest.fixture
def tiny_model() -> GPT2:
    """1-layer GPT-2 small enough that forward passes take milliseconds."""
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    cfg = GPT2Config(vocab_size=50, n_embd=8, block_size=4, n_layer=1, n_head=1)
    m = GPT2(cfg)
    # Skip torch.compile in tests — avoids compilation overhead
    m.supports_compile = lambda device_type: False  # type: ignore[method-assign]
    return m


class _FakeLoader:
    """Minimal data loader — returns random (x, y) of fixed shape."""

    def __init__(
        self,
        B: int = 1,
        T: int = 4,
        vocab_size: int = 50,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.B = B
        self.T = T
        self.tokenizer_name = tokenizer_name
        self._vocab = vocab_size

    def reset(self) -> None:
        pass

    def advance(self, tokens: int) -> None:
        pass

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, self._vocab, (self.B, self.T))
        y = torch.randint(0, self._vocab, (self.B, self.T))
        return x, y


def _make_trainer(
    tmp_path: Path,
    model: GPT2,
    cfg: TrainConfig,
    train_loader: Any = None,
    val_loader: Any = None,
) -> Trainer:
    train_loader = train_loader or _FakeLoader()
    val_loader = val_loader or _FakeLoader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=cfg,
        run_config=_run_config(),
    )


# ---------------------------------------------------------------------------
# Trainer scheduling tests
# ---------------------------------------------------------------------------


def test_trainer_runs_expected_evals_on_schedule(tmp_path: Path, tiny_model: GPT2) -> None:
    """eval_loss must fire every <eval_loss_every> steps, not more, not less.

    max_steps=6, eval_every=3:
      in-loop  → steps 0, 3
      final    → last_step=5, 5%3≠0 → fires once more
      total    = 3 calls
    """
    cfg = _train_config(tmp_path, max_steps=6, eval_loss_every=3, checkpoint_every=1000)
    trainer = _make_trainer(tmp_path, tiny_model, cfg)

    trainer._eval_loss = MagicMock(return_value=0.5)  # type: ignore[method-assign]
    trainer._save_checkpoint = MagicMock()  # type: ignore[method-assign]
    trainer.train()

    assert trainer._eval_loss.call_count == 3  # type: ignore[union-attr]


def test_trainer_runs_final_eval_when_last_step_not_aligned(
    tmp_path: Path, tiny_model: GPT2
) -> None:
    """Both eval_loss and checkpoint must be called at the very last step when
    max_steps-1 is not aligned with their respective periods.

    max_steps=5, eval_every=3, checkpoint_every=3:
      in-loop  → step 0 (eval+ckpt), step 3 (eval+ckpt)
      final    → last_step=4, 4%3≠0 → eval+ckpt fire again
      totals   = 3 eval calls, 3 checkpoint calls
    """
    cfg = _train_config(tmp_path, max_steps=5, eval_loss_every=3, checkpoint_every=3)
    trainer = _make_trainer(tmp_path, tiny_model, cfg)

    trainer._eval_loss = MagicMock(return_value=0.5)  # type: ignore[method-assign]
    trainer._save_checkpoint = MagicMock()  # type: ignore[method-assign]
    trainer.train()

    assert trainer._eval_loss.call_count == 3  # type: ignore[union-attr]
    assert trainer._save_checkpoint.call_count == 3  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Trainer resume tests
# ---------------------------------------------------------------------------


def test_trainer_resume_pretrain_loader_advance_matches_checkpoint(
    tmp_path: Path, tiny_model: GPT2
) -> None:
    """On a normal resume the loader must advance by
    checkpoint_step × total_batch_size tokens, and self.step must be
    checkpoint_step + 1."""
    RESUME_STEP = 7
    TOTAL_BATCH = 4

    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

    # Write a real checkpoint so torch.load works without mocking
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "step": RESUME_STEP,
            "model_dict": tiny_model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    train_loader = _FakeLoader()
    advance_log: list[int] = []
    train_loader.advance = lambda tokens: advance_log.append(tokens)  # type: ignore[method-assign]

    cfg = _train_config(
        tmp_path,
        max_steps=RESUME_STEP + 5,
        eval_loss_every=1000,
        total_batch_size=TOTAL_BATCH,
        resume_from=ckpt_path,
        resume_weights_only=False,
    )
    trainer = _make_trainer(tmp_path, tiny_model, cfg, train_loader=train_loader)

    assert trainer.step == RESUME_STEP + 1
    assert advance_log == [RESUME_STEP * TOTAL_BATCH]


def test_trainer_resume_sft_loader_advance_matches_checkpoint_semantics(
    tmp_path: Path, tiny_model: GPT2
) -> None:
    """With resume_weights_only=True the step is reset to 0 and advance() is
    never called — SFT training always starts from the beginning of the data."""
    RESUME_STEP = 50

    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    ckpt_path = tmp_path / "ckpt_sft.pt"
    torch.save(
        {
            "step": RESUME_STEP,
            "model_dict": tiny_model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    train_loader = _FakeLoader()
    advance_log: list[int] = []
    train_loader.advance = lambda tokens: advance_log.append(tokens)  # type: ignore[method-assign]

    cfg = _train_config(
        tmp_path,
        max_steps=RESUME_STEP + 5,
        eval_loss_every=1000,
        resume_from=ckpt_path,
        resume_weights_only=True,
    )
    trainer = _make_trainer(tmp_path, tiny_model, cfg, train_loader=train_loader)

    assert trainer.step == 0  # reset, not resumed
    assert advance_log == []  # advance never called
